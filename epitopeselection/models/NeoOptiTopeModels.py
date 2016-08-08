"""
An extension of OptiTope for personalized epitope selection
in the context of cancer immunotherapy development

Multiple objective functions are available depending on
the provided input:

Simple objective function:
1) max \sum_{e \in E} x_e \cdot (\sum_{g \in G(e)} log(a_g)) \cdot \sum_{h \in H} a_h \cdot i_{e,a}

Objective with distance-to-self d_e (assumed to be in the rage [0,1])
2) max \sum_{e \in E} x_e \cdot (\sum_{g \in G(e)} log(a_g)) \cdot \sum_{h \in H} a_h \cdot i_{e,a} \cdot d_{e,h}

Objective with distance-to-self and uncertainty associated with immunogenicity prediction.
This model assumes that the each epitope can be assumed as iid Bernouli distributed random variable.
Hence the Overall objective is the Expacation value of a Poissan-Binomial distribution and the uncertainty
of the objective can be measured by the variance of the Poissan-Binomial distribution. The selection is performed
by a Pareto optimization and will return all Pareto optimal solutions of Immunogenicity and associated Risk.
3) max \sum_{e \in E} x_e \cdot (\sum_{g \in G(e)} log(a_g)) \cdot \sum_{h \in H} a_h \cdot i_{e,a} \cdot d_{e,h}
   min \sum_{e \in E} x_e \sum_{h \in H} sigma_{e,h}
"""

import copy
import itertools as itr
import math

from pyomo.environ import ConcreteModel, Set, Param, Var, Constraint, PositiveIntegers, \
                          Binary, NonNegativeIntegers, Objective, maximize, minimize, NonNegativeReals, \
                          NonPositiveIntegers
from pyomo.opt import SolverFactory, TerminationCondition

from Fred2.Core import EpitopePredictionResult


class NeoOptiTope(object):

  def __init__(self, results, threshold=None, distance={}, expression={}, uncertainty={}, overlap=0, k=10, k_taa=0,
               solver="glpk", verbosity=0):
        """
        :param result: Epitope prediction result object from which the epitope selection should be performed
        :type result: :class:`~Fred2.Core.Result.EpitopePredictionResult`
        :param dict(str,float) threshold: A dictionary scoring the binding thresholds for each HLA
                                          :class:`~Fred2.Core.Allele.Allele` key = allele name; value = the threshold
        :param dict((str,str),float) distance: A dicitionary with key: (peptide sequence, HLA name)
                                               and value the distance2self
        :param dict(str, float) expression: A dictionary with key: gene ID, and value: Gene expression
                                            in FPKM/RPKM or TPM
        :param dict((str,str),float) uncertainty: A dictionary with key (peptide seq, HLA name), and value the
                                                  associated uncertainty of the immunogenicity prediction
        :param int k: The number of epitopes to select
        :param int k_taa: The number of TAA epitopes to select
        :param str solver: The solver to be used (default glpk)
        :param int verbosity: Integer defining whether additional debugg prints are made >0 => debug mode
        """
        #check input data
        if not isinstance(results, EpitopePredictionResult):
            raise ValueError("first input parameter is not of type EpitopePredictionResult")

        _alleles = copy.deepcopy(results.columns.values.tolist())

        #Generate abundance dictionary of HLA alleles default is 2.0 as values will be log2 transformed
        probs = {a.name:2.0 if a.get_metadata("abundance", only_first=True) is None else
                 a.get_metadata("abundance", only_first=True) for a in _alleles}
        if verbosity:
            for a in _alleles:
                print a.name, a.prob

        #start constructing model
        self.__solver = SolverFactory(solver)
        self.__verbosity = verbosity
        self.__changed = True
        self.__alleleProb = _alleles
        self.__k = k
        self.__k_taa = k_taa
        self.__result = None
        self.__thresh = {} if threshold is None else threshold
        self.overlap=overlap
        # Variable, Set and Parameter preparation
        alleles_I = {}
        variations = []
        epi_var = {}
        imm = {}
        peps = {}
        taa = []
        var_epi = {}
        cons = {}

        #unstack multiindex df to get normal df based on first prediction method
        #and filter for binding epitopes
        method = results.index.values[0][1]
        res_df = results.xs(results.index.values[0][1], level="Method")
        res_df = res_df[res_df.apply(lambda x: any(x[a] > self.__thresh.get(a.name, -float("inf"))
                                                   for a in res_df.columns), axis=1)]

        #transform scores to 1-log50k(IC50) scores if neccassary
        #and generate mapping dictionaries for Set definitions
        for tup in res_df.itertuples():
            p = tup[0]
            seq = str(p)
            peps[seq] = p
            if p.get_metadata("taa",only_first=True):
                taa.append(seq)
            for a, s in itr.izip(res_df.columns, tup[1:]):
                if method in ["smm", "smmpmbec", "arb", "comblibsidney"]:
                    try:
                        thr = min(1., max(0.0, 1.0 - math.log(self.__thresh.get(a.name),
                                                      50000))) if a.name in self.__thresh else -float("inf")
                    except:
                        thr = 0

                    if s >= thr:
                        alleles_I.setdefault(a.name, set()).add(seq)
                    imm[seq, a.name] = min(1., max(0.0, 1.0 - math.log(s, 50000)))
                else:
                    if s > self.__thresh.get(a.name, -float("inf")):
                        alleles_I.setdefault(a.name, set()).add(seq)
                    imm[seq, a.name] = s

            prots = set(pr for pr in p.get_all_proteins())
            cons[seq] = len(prots)
            for prot in prots:
                variations.append(prot.gene_id)
                epi_var.setdefault(prot.gene_id, set()).add(seq)
                var_epi.setdefault(str(seq), set()).add(prot.gene_id)
        self.__peptideSet = peps

        #calculate conservation
        variations = set(variations)
        total = len(variations)
        for e, v in cons.iteritems():
            try:
                cons[e] = v / total
            except ZeroDivisionError:
                cons[e] = 1
        model = ConcreteModel()

        ######################################
        #
        # MODEL DEFINITIONS
        #
        ######################################
        #set definition
        model.Q = Set(initialize=variations)
        model.E = Set(initialize=set(peps.keys()))
        model.TAA = Set(initialize=set(taa))
        model.A = Set(initialize=alleles_I.keys())
        model.G = Set(model.E, initialize=lambda model, e: var_epi[e])
        model.E_var = Set(model.Q, initialize=lambda mode, v: epi_var[v])
        model.A_I = Set(model.A, initialize=lambda model, a: alleles_I[a])
        if overlap > 0:
            def longest_common_substring(model):
                result = []
                for s1,s2 in itr.combinations(model.E,2):
                    if s1 != s2:
                        if s1 in s2 or s2 in s1:
                            result.append((s1,s2))
                        m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
                        longest, x_longest = 0, 0
                        for x in xrange(1, 1 + len(s1)):
                            for y in xrange(1, 1 + len(s2)):
                                if s1[x - 1] == s2[y - 1]:
                                    m[x][y] = m[x - 1][y - 1] + 1
                                    if m[x][y] > longest:
                                        longest = m[x][y]
                                        x_longest = x
                                else:
                                    m[x][y] = 0
                        if len(s1[x_longest - longest: x_longest]) >= overlap:
                            result.append((s1,s2))
                return set(result)
            model.O = Set(dimen=2, initialize=longest_common_substring)

        #parameter definition
        model.k = Param(initialize=self.__k, within=PositiveIntegers, mutable=True)
        model.k_taa = Param(initialize=self.__k_taa, within=NonPositiveIntegers, mutable=True)
        model.p = Param(model.A, initialize=lambda model, a: max(0, math.log(probs[a]+0.001,2)))
        model.c = Param(model.E, initialize=lambda model, e: cons[e],mutable=True)
        model.sigma = Param (model. E, model.A, initialize=lambda model, e, a: uncertainty.get((e,a), 0))
        model.i = Param(model.E, model.A, initialize=lambda model, e, a: imm[e, a])
        model.t_allele = Param(initialize=0, within=NonNegativeIntegers, mutable=True)
        model.t_var = Param(initialize=0, within=NonNegativeIntegers, mutable=True)
        model.t_c = Param(initialize=0.0, within=NonNegativeReals, mutable=True)
        model.abd = Param(model.Q, initialize=lambda model, g: max(0, math.log(expression.get(g, 2)+0.001, 2)))
        model.d = Param(model.E, model.A, initialize=lambda model, e, a: distance.get((e,a), 1))
        model.eps1 = Param(initialize=1e6, mutable=True)
        model.eps2 = Param(initialize=1e6, mutable=True)

        # Variable Definition
        model.x = Var(model.E, within=Binary)
        model.y = Var(model.A, within=Binary)
        model.z = Var(model.Q, within=Binary)

        # Objective definition
        model.Obj1 = Objective(
            rule=lambda model: -sum(model.x[e] * sum(model.abd[g] for g in model.G[e])
                             * sum(model.p[a] * model.i[e, a] * model.d[e, a] for a in model.A) for e in model.E),
            sense=minimize)
        model.Obj2 = Objective(
            rule=lambda model: sum(model.x[e]*sum(model.sigma[e,a] for a in model.A) for e in model.E),
            sense=minimize)

        #Constraints
        #Obligatory Constraint (number of selected epitopes)
        model.NofSelectedEpitopesCov1 = Constraint(rule=lambda model: sum(model.x[e] for e in model.E) >= model.k)
        model.NofSelectedEpitopesCov2 = Constraint(rule=lambda model: sum(model.x[e] for e in model.E) <= model.k)
        model.NofSelectedTAACov = Constraint(rule=lambda model: sum(model.x[e] for e in model.TAA) <= model.k_taa)

        #optional constraints (in basic model they are disabled)
        model.IsAlleleCovConst = Constraint(model.A,
                                            rule=lambda model, a: sum(model.x[e] for e in model.A_I[a]) >= model.y[a])

        model.MinAlleleCovConst = Constraint(rule=lambda model: sum(model.y[a] for a in model.A) >= model.t_allele)

        model.IsAntigenCovConst = Constraint(model.Q,
                                             rule=lambda model, q: sum(model.x[e] for e in model.E_var[q]) >= model.z[q])
        model.MinAntigenCovConst = Constraint(rule=lambda model: sum(model.z[q] for q in model.Q) >= model.t_var)

        model.EpitopeConsConst = Constraint(model.E,
                                            rule=lambda model, e: (1 - model.c[e]) * model.x[e] <= 1 - model.t_c)

        if overlap > 0:
            model.OverlappingConstraint = Constraint(model.O, rule=lambda model, e1, e2: model.x[e1]+model.x[e2] <= 1)

        #Constraints for Pareto optimization
        model.ImmConst = Constraint(rule=lambda model: sum(model.x[e] * sum(model.abd[g] for g in model.G[e])
                                                       * sum(model.p[a] * model.i[e, a] * model.d[e, a]
                                                       for a in model.A) for e in model.E) <= model.eps1)
        model.UncertaintyConst = Constraint(rule=lambda model:sum(model.x[e]*sum(model.sigma[e,a]
                                                                                 for a in model.A)
                                                                  for e in model.E) <= model.eps2)
        self.__objectives = [model.Obj1, model.Obj2]
        self.__constraints = [model.UncertaintyConst, model.ImmConst]
        self.__epsilons = [model.eps2, model.eps1]

        #generate instance
        self.instance = model
        if self.__verbosity > 0:
            print "MODEL INSTANCE"
            self.instance.pprint()

        #constraints
        self.instance.Obj2.deactivate()
        self.instance.ImmConst.deactivate()
        self.instance.UncertaintyConst.deactivate()
        self.instance.IsAlleleCovConst.deactivate()
        self.instance.MinAlleleCovConst.deactivate()
        self.instance.IsAntigenCovConst.deactivate()
        self.instance.MinAntigenCovConst.deactivate()
        self.instance.EpitopeConsConst.deactivate()

  def set_k(self, k):
        """
            Sets the number of epitopes to select

            :param int k: The number of epitopes
            :raises ValueError: If the input variable is not in the same domain as the parameter
        """
        tmp = self.instance.k.value
        try:
            getattr(self.instance, str(self.instance.k)).set_value(int(k))
            self.__changed = True
        except ValueError:
            self.__changed = False
            getattr(self.instance, str(self.instance.k)).set_value(int(tmp))
            raise ValueError('set_k', 'An error has occurred during setting parameter k. Please check if k is integer.')

  def set_k_taa(self, k):
        """
            Sets the number of epitopes to select

            :param int k: The number of epitopes
            :raises ValueError: If the input variable is not in the same domain as the parameter
        """
        tmp = self.instance.k_taa.value
        try:
            getattr(self.instance, str(self.instance.k_taa)).set_value(int(k))
            self.__changed = True
        except ValueError:
            self.__changed = False
            getattr(self.instance, str(self.instance.k_taa)).set_value(int(tmp))
            raise ValueError('set_k', 'An error has occurred during setting parameter k_taa. Please check if k is integer.')


  def activate_allele_coverage_const(self, minCoverage):
        """
            Enables the allele coverage constraint

            :param float minCoverage: Percentage of alleles which have to be covered [0,1]
            :raises ValueError: If the input variable is not in the same domain as the parameter
        """
        # parameter
        mc = self.instance.t_allele.value

        try:
            #getattr(self.instance, str(self.instance.t_allele)).set_value(max(1,int(len(self.__alleleProb) * minCoverage)))
            getattr(self.instance, str(self.instance.t_allele))[None] = max(1,int(len(self.__alleleProb) * minCoverage))
            #variables
            #constraints
            self.instance.IsAlleleCovConst.activate()
            self.instance.MinAlleleCovConst.activate()
            self.__changed = True
        except ValueError:
            getattr(self.instance, str(self.instance.t_allele)).set_value(mc)
            self.__changed = False
            raise ValueError(
                'activate_allele_coverage_const","An error occurred during activation of of the allele coverage constraint. ' +
                'Please check your specified minimum coverage parameter to be in the range of 0.0 and 1.0.')

  def deactivate_allele_coverage_const(self):
        """
            Deactivates the allele coverage constraint
        """

        # parameter
        self.__changed = True

        #constraints
        self.instance.IsAlleleCovConst.deactivate()
        self.instance.MinAlleleCovConst.deactivate()

  def activate_antigen_coverage_const(self, t_var):
        """
            Activates the variation coverage constraint

            :param int t_var: The number of epitopes which have to come from each variation
            :raises ValueError: If the input variable is not in the same domain as the parameter

        """
        tmp = self.instance.t_var.value
        try:
            getattr(self.instance, str(self.instance.t_var)).set_value(max(1,int(len(self.instance.Q)*t_var)))
            self.instance.IsAntigenCovConst.activate()
            self.instance.MinAntigenCovConst.activate()
            self.__changed = True
        except ValueError:
            getattr(self.instance, str(self.instance.t_var)).set_value(int(tmp))
            self.instance.IsAntigenCovConst.deactivate()
            self.instance.MinAntigenCovConst.deactivate()
            self.__changed = False
            raise ValueError("activate_antigen_coverage_const",
                            "An error has occurred during activation of the coverage constraint. Please make sure your input is an integer.")

  def deactivate_antigen_coverage_const(self):
        """
            Deactivates the variation coverage constraint
        """
        self.__changed = True
        self.instance.IsAntigenCovConst.deactivate()
        self.instance.MinAntigenCovConst.deactivate()

  def activate_epitope_conservation_const(self, t_c, conservation=None):
        """
            Activates the epitope conservation constraint

            :param float t_c: The percentage of conservation an epitope has to have [0.0,1.0].
            :param: conservation: A dict with key=:class:`~Fred2.Core.Peptide.Peptide` specifying a different
                                  conservation score for each :class:`~Fred2.Core.Peptide.Peptide`
            :type conservation: dict(:class:`~Fred2.Core.Peptide.Peptide`,float)
            :raises ValueError: If the input variable is not in the same domain as the parameter
        """
        if t_c < 0 or t_c > 1:
            raise ValueError("activate_epitope_conservation_const",
                            "The conservation threshold is out of its numerical bound. It has to be between 0.0 and 1.0.")

        self.__changed = True
        getattr(self.instance, str(self.instance.t_c)).set_value(float(t_c))
        if conservation is not None:
            for e in self.instance.E:
                if e in conservation:
                    getattr(self.instance, str(self.instance.c))[e] = conservation[e]
                else:
                    getattr(self.instance, str(self.instance.c))[e] = 0.0

        self.instance.EpitopeConsConst.activate()

  def deactivate_epitope_conservation_const(self):
        """
            Deactivates epitope conservation constraint
        """
        self.__changed = True
        self.instance.EpitopeConsConst.deactivate()

  def __leximin(self, eps=1e6, order=(0,1), options={}):
        """
        The lexmin operation to find a pareto optimal point

            :param eps: Current upper boind on Objective
            :param order: the oder of the lexmin operation
            :param options: dictionary of solver options
            :return: The two objective values and the pareot-optimal assembly as triple
            :rtype: tuple(float,float,list(Peptide))
        """
        objs = [0,0]
        self.__objectives[order[0]].activate()
        self.__objectives[order[1]].deactivate()
        self.__constraints[order[0]].activate()
        self.__constraints[order[1]].deactivate()

        getattr(self.instance, str(self.__epsilons[order[0]])).set_value(float(eps))

        res = self.__solver.solve(self.instance, options=options)
        self.instance.solutions.load_from(res)
        objs[order[0]] = self.__objectives[order[0]].expr()
        if self.__verbosity > 0:
            res.write(num=1)
            print "Objective {nof_obj}:{value}".format(nof_obj=order[0],value=objs[order[0]])

        self.__objectives[order[1]].activate()
        self.__objectives[order[0]].deactivate()
        self.__constraints[order[1]].activate()
        self.__constraints[order[0]].deactivate()

        getattr(self.instance, str(self.__epsilons[order[1]])).set_value(objs[order[0]])

        res = self.__solver.solve(self.instance, options=options)
        self.instance.solutions.load_from(res)
        objs[order[1]] = self.__objectives[order[1]].expr()
        if self.__verbosity > 0:
            res.write(num=1)
            print "Objective {nof_obj}:{value}".format(nof_obj=order[1],value=objs[order[1]])

        return objs[0], objs[1], [self.__peptideSet[x] for x in self.instance.x
                                                            if 0.8 <= self.instance.x[x].value <= 1.2]

  def solve(self, pareto=False, options=None, rel_tol=1e-09, abs_tol=0.0001):
        """
            Invokes the selected solver and solves the problem

            :param bool pareto: Invoces pareto optimization (makes only sense if uncertainty was specified)
            :param dict(str,str) options: A dictionary of solver specific options as keys and their parameters as values
            :param float rel_tol: relative floating point similarity tolerance
            :param float abs_tol: absolute floating point similarity tolerance
            :return Returns the optimal epitopes as list of :class:`~Fred2.Core.Peptide.Peptide` objectives
            :rtype: list(:class:`~Fred2.Core.Peptide.Peptide`)
            :raise RuntimeError: If the solver raised a problem or the solver is not accessible via the PATH
                                 environmental variable.
        """
        options = dict() if options is None else options

        if self.__changed:
            try:
                if not pareto:
                    res = self.__solver.solve(self.instance, options=options)
                    self.instance.solutions.load_from(res)
                    if self.__verbosity > 0:
                        res.write(num=1)

                    if res.solver.termination_condition != TerminationCondition.optimal:
                        raise RuntimeError("Could not solve problem - " + str(res.Solution.status) + ". Please check your settings")

                    self.__result = [(self.instance.Obj1.expr(),None, [self.__peptideSet[x] for x in self.instance.x
                                                                       if 0.8 <= self.instance.x[x].value <= 1.2])]
                    #self.__result.log_metadata("obj", res.Solution.Objective.Value)

                    self.__changed = False
                    return self.__result

                else:
                    def __isclose(a, b):
                        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

                    zT = self.__leximin(options=options)
                    pareto_front = [zT]
                    zB = self.__leximin(order=(1,0), options=options)
                    pareto_front.append(zB)
                    while True:
                        zT = self.__leximin(eps=zT[1]-abs_tol, options=options)
                        if __isclose(zT[1], zB[1]):
                            self.__changed = False
                            self.__result = sorted(pareto_front)
                            return self.__result

            except Exception as e:
                print e
                raise RuntimeError("solve",
                                "An Error has occurred during solving. Please check your settings and if the solver is registered in PATH environment variable.")
        else:
            return self.__result

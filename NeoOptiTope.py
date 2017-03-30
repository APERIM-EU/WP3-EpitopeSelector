#!/usr/bin/env python

"""
usage: NeoOptiTope.py [-h] -i INPUT [-imm IMMUNOGENICITY] [-d DISTANCE]
                      [-u UNCERTAINTY] [-taa TAA] -a ALLELES [-k K]
                      [-ktaa KTAA] [-incl INCLUDE] [-excl EXCLUDE]
                      [-te THRESHOLD_EPITOPE] [-td THRESHOLD_DISTANCE] -o
                      OUTPUT [-s SOLVER] [-c_al CONS_ALLELE]
                      [-c_a CONS_ANTIGEN] [-c_o CONS_OVERLAP] [-r]
                      [-opt OPTIONS]

Epitope Selection for personalized vaccine design.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Peptide with immunogenicity file (from epitope
                        prediction)
  -imm IMMUNOGENICITY, --immunogenicity IMMUNOGENICITY
                        Column name of peptide immunogenicity
  -d DISTANCE, --distance DISTANCE
                        Column name of distance-to-self calculation
  -u UNCERTAINTY, --uncertainty UNCERTAINTY
                        Column name of prediction uncertainty
  -taa TAA, --taa TAA   Column name specifying whether the peptide is a TAA or
                        TSA. (if not specified all peptides are assumed to be
                        TSAs)
  -a ALLELES, --alleles ALLELES
                        Allele file with expression values
  -k K, --k K           Specifies the number of epitopes to select
  -ktaa KTAA, --ktaa KTAA
                        Specifies the number of TAA epitopes that are allowed
                        to select
  -incl INCLUDE, --include INCLUDE
                        Epitope file with epitopes which should be included
  -excl EXCLUDE, --exclude EXCLUDE
                        Epitope file with epitopes which should be excluded
  -te THRESHOLD_EPITOPE, --threshold_epitope THRESHOLD_EPITOPE
                        Specifies the binding/immunogenicity threshold for all
                        alleles
  -td THRESHOLD_DISTANCE, --threshold_distance THRESHOLD_DISTANCE
                        Specifies the distance-to-self threshold for all
                        alleles
  -o OUTPUT, --output OUTPUT
                        Specifies the output file. Results will be written to
                        CSV
  -s SOLVER, --solver SOLVER
                        Specifies the ILP solver
  -c_al CONS_ALLELE, --cons_allele CONS_ALLELE
                        Activates allele coverage constraint with specified
                        threshold ]0,1]
  -c_a CONS_ANTIGEN, --cons_antigen CONS_ANTIGEN
                        Activates antigen coverage constraint with specified
                        threshold ]0,1]
  -c_o CONS_OVERLAP, --cons_overlap CONS_OVERLAP
                        Activates epitope overlapping constraint with
                        specified threshold
  -r, --rank            Compute selection on rank input
  -opt OPTIONS, --options OPTIONS
                        String of the form key1=value1,key2=value2 that
                        specify solver specific options (This will not be
                        check for correctness)
"""

import sys
import pandas
import argparse
import csv
import logging
import math
import os
import datetime

import pandas as pd
import itertools as itr
from Fred2.Core import Allele, Peptide, Protein, EpitopePredictionResult
from models.NeoOptiTopeModels import NeoOptiTope


def read_epitope_input(args, alleles, exclude):
    """
    reads in epitope files generated by NGSAnalyzer+ImmogenicityPredictor

    Header NGSAnalyzer:
        mutation - position of the mutation in the reference genome (currently hg19); format:
                   chromosome_position; the position is zero-based
        gene - gene affected by the mutation
        transcript - transcript affected by the mutation (UCSC known genes transcript ID)
        transcript_expression - expression in RPKM/FPKM of the affected transcript
        neopeptide - peptide resulting from the mutation in the given transcript
        length_of_neopeptide - length of the neopeptide
        HLA - HLA used for the binding prediction of the neopeptide
        HLA_class1_binding_prediction - predicted binding affinity (currently rank score of IEDB consensus tool)

    Header ImmunogenicityPredictor:
        immunogenicity - predicted immunogenicity for specific HLA allele in column
        distance - distance-to-self estimation specific for HLA allele in column
        [uncertainty] - If immunopredictor can estimate prediction uncertainty

    :param args: Input arguments
    :param alleles: HLA alleles
    :param exlude: excluded peptides
    :return: df_epitope - EpitopePredictionResult
             distance - dict(pep_string, float)
             expression - dict(gene_id, float)
             uncertainty - dict(pep_string, float)
             pep_to_mutation - dict(pep_string, mutation_string)
    """
    distance = {}
    uncertainty = {}
    expression = {}
    seq_to_pep = {}
    gene_to_prot = {}
    hla_to_allele = {a.name:a for a in alleles}
    df_pred = {a:{} for a in alleles}
    pep_to_mutation = {}

    with open(args.input, "rU") as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            seq = row["neopeptide"]
            if (exclude is None or seq not in exclude) and len(seq) > 0:
                if seq in seq_to_pep:
                    pep = seq_to_pep[seq]
                else:
                    pep = Peptide(seq.upper())
                    seq_to_pep[seq] = pep
                try:
                	allele = hla_to_allele[row["HLA"].replace("HLA-","")]
                except:
                    logging.warning(
                            "HLA {allele} was not contained in the provided allele file. Please check your input.".format(
                                allele=row["HLA"]))
                    continue
                gene = row["gene"]
                if gene in gene_to_prot:
                    prot = gene_to_prot[gene]
                else:
                    prot = Protein("", gene_id=gene, transcript_id=gene)
                    gene_to_prot[gene] = prot

                pep.proteins[prot.transcript_id]=prot
                pep.proteinPos[prot.transcript_id].append(0)
                pep_to_mutation.setdefault(seq, []).append(row["mutation"])
                expression.setdefault(gene,[]).append(float(row["transcript_expression"]))

                if args.rank is not None:
                	df_pred[allele][pep] = max(0., 1.
                                            - float(row[args.immunogenicity])/100.0)
                else:
                	df_pred[allele][pep] = max(0., 1.
                                            - math.log(float(row[args.immunogenicity]),
                                                50000)) if args.immunogenicity == "HLA_class1_binding_prediction" else float(
                        row[args.immunogenicity])

                if args.distance is not None:
                    distance[(seq,allele.name)] = float(row[args.distance])

                if args.uncertainty is not None:
                    uncertainty[(seq,allele.name)] = float(row[args.uncertainty])

                if args.taa is not None:
                    pep.log_metadata("taa", row[args.taa].upper() == "TAA" )

    expression = {k:max(v) for k,v in expression.iteritems()}
    df_result = EpitopePredictionResult.from_dict(df_pred)
    df_result.index = pandas.MultiIndex.from_tuples([tuple((i, "custom")) for i in df_result.index],
                                                        names=['Seq', 'Method'])

    return df_result, distance, expression, uncertainty, pep_to_mutation


def read_hla_input(hla_file):
    """
    reads in the hla file
    header are defined as:

    A1 - first HLA-A allele in 4-digit notation
    A2 - second HLA-A allele in 4-digit notation
    B1 - first HLA-B allele in 4-digit notation
    B2 - second HLA-B allele in 4-digit notation
    C1 - first HLA-C allele in 4-digit notation
    C2 - second HLA-C allele in 4-digit notation
    A_expression - expression of HLA A gene
    B_expression - expression of HLA B gene
    C_expression - expression of HLA C gene

    :param hla_file:
    :return: list(Allele)
    """
    alleles = []

    with open(hla_file, "rU") as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            for n, hla in itr.product([1,2],["A","B","C"]):
                a = Allele(row[hla+str(n)])
                a.log_metadata("abundance",float(row[hla+"_expression"]))
                alleles.append(a)

    return alleles

def read_incl_input(incl_file):
    """
    reads in the include file

    :param in_file:
    :return: list(included_epitopes)
    """
    included_epitopes = []
    if not incl_file is None:
        with open(incl_file, "rU") as f:
            included_epitopes = f.read().splitlines()
        return included_epitopes
    else:
        return None


def read_excl_input(excl_file):
    """
    reads in the exclude file

    :param excl_file:
    :return: list(excluded_epitopes)
    """
    excluded_epitopes = []

    if not excl_file is None:
        with open(excl_file, "rU") as f:
            excluded_epitopes = f.read().splitlines()
        return excluded_epitopes
    else:
        return None

def main():
    parser = argparse.ArgumentParser(
                      description="Epitope Selection for personalized vaccine design.",

    )
    parser.add_argument("-i","--input",
               required=True,
               type=str,
               help="Peptide with immunogenicity file (from epitope prediction)",
    )

    parser.add_argument("-imm","--immunogenicity",
               required=False,
               default="HLA_class1_binding_prediction",
               type=str,
               help="Column name of peptide immunogenicity",
    )

    parser.add_argument("-d","--distance",
               required=False,
               default=None,
               type=str,
               help="Column name of distance-to-self calculation",
    )

    parser.add_argument("-u","--uncertainty",
               required=False,
               default=None,
               type=str,
               help="Column name of prediction uncertainty",
    )

    parser.add_argument("-taa","--taa",
               required=False,
               type=str,
               default=None,
               help="Column name specifying whether the peptide is a TAA or TSA. (if not specified all peptides are assumed to be TSAs)"
    )

    parser.add_argument("-a","--alleles",
               required=True,
               default=None,
               type=str,
               help="Allele file with expression values",
    )

    parser.add_argument("-k","--k",
               required=False,
               type=int,
               default=10,
               help="Specifies the number of epitopes to select",
    )

    parser.add_argument("-ktaa","--ktaa",
               required=False,
               type=int,
               default=0,
               help="Specifies the number of TAA epitopes that are allowed to select",
    )

    parser.add_argument("-incl","--include",
                required=False,
                default=None,
                type=str,
                help="Epitope file with epitopes which should be included",
     )

    parser.add_argument("-excl","--exclude",
                required=False,
                default=None,
                type=str,
                help="Epitope file with epitopes which should be excluded",
     )

    parser.add_argument("-te", "--threshold_epitope",
               type=float,
               default=0.,
               help="Specifies the binding/immunogenicity threshold for all alleles",
    )

    parser.add_argument("-td", "--threshold_distance",
               type=float,
               default=0.,
               help="Specifies the distance-to-self threshold for all alleles",
    )
    parser.add_argument("-o", "--output",
               required=True,
               type=str,
               help="Specifies the output file. Results will be written to CSV",
    )

    parser.add_argument("-s","--solver",
               type=str,
               default="cbc",
               help="Specifies the ILP solver")

    parser.add_argument("-c_al", "--cons_allele",
               required=False,
               type=float,
               default=0.0,
               help="Activates allele coverage constraint with specified threshold ]0,1]",
    )

    parser.add_argument("-c_a", "--cons_antigen",
               required=False,
               type=float,
               default=0.0,
               help="Activates antigen coverage constraint with specified threshold ]0,1]",
    )

    parser.add_argument("-c_o", "--cons_overlap",
               required=False,
               type=int,
               default=0,
               help="Activates epitope overlapping constraint with specified threshold",
    )

    parser.add_argument('-r', "--rank", 
    	       action='store_true',
               help="Compute selection on rank input"
    )

    parser.add_argument("-opt", "--options",
               required=False,
               type=str,
               default="",
               help="String of the form key1=value1,key2=value2 that specify solver \
                     specific options (This will not be check for correctness)",
    )

    args = parser.parse_args()
    hlas = read_hla_input(args.alleles)
    exclude = read_excl_input(args.exclude)
    included = read_incl_input(args.include)

    df_epitope, distance, expression, uncertainty, pep_to_mutation = read_epitope_input(args, hlas, exclude)
    thresh = {a.name:args.threshold_epitope for a in hlas}

    options = {k.strip():v.strip() for s in args.options.split(",")
               for k,v in s.split("=")} if "=" in args.options else {}

    model = NeoOptiTope(df_epitope,
                        k=args.k,
                        k_taa=args.ktaa,
                        threshold=thresh,
                        distance=distance,
                        dist_threshold=float(args.threshold_distance),
                        expression=expression,
                        uncertainty=uncertainty,
                        overlap=args.cons_overlap,
                        include=included,
                        solver=args.solver,
                        )

    if args.cons_allele > 0:
        model.activate_allele_coverage_const(args.cons_allele)

    if args.cons_antigen > 0:
        model.activate_antigen_coverage_const(args.cons_antigen)

    output = model.solve(options=options) if args.uncertainty is None else model.solve(pareto=True, options=options)

    # generate output:
    this_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    tissues = pd.read_csv(os.path.join(this_dir, 'data/rna_tissue.csv'), index_col=[1,2])
    instance = model.instance
    tis = set(tissues.index.levels[1])
    with open(args.output, "w") as f:
        f.write("#GS\n")
        f.write("date\timm_column\tinput_peptide_file\tinput_HLA_file\n{date}\t{imm}\t{pep_in}\t{HLA_in}\n\n".format(
            date=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M"),
            imm=args.immunogenicity,
            pep_in=args.input,
            HLA_in=args.alleles
        ))
        for i, (obj1,obj2,es) in enumerate(output):
            # peptide Section
            cov_anti = []
            for an in instance.Q:
                for e in es:
                    if e in instance.E_var[an].value:
                        cov_anti.append(an)
            cov_anti = set(cov_anti)

            cov_als = []
            res_set = set(es)
            locus = {}
            for a in instance.A:
                eps_of_all_i = list(instance.A_I[a])
                if res_set.intersection(set(eps_of_all_i)):
                    cov_als.append(a)
                    locus.setdefault(str(a).split("*")[0], set()).add(a)
            cov_als = set(cov_als)
            f.write("#CS-{number}\n".format(number=i))
            f.write("nof_epitopes\tnof_taa_epitopes\tthreshold_epitope\tthreshold_distance\tantigen_const\thla_const\toverlap_const\tdistance2self\tuncertainty\timmunogenicity\trisk\tcovered_hlas\tcovered_antigens\n")
            f.write("{nof_epitopes}\t{nof_taa_epitopes}\t{threshold}\t{threshold_distance}\t{antigen_const}\t{hla_const}\t{overlap_const}\t{distance2self}\t{uncertainty}\t{immunogenicity}\t{risk}\t{covered_hlas}\t{covered_antigens}\n".format(
                nof_epitopes=instance.k.value,
                nof_taa_epitopes=instance.k_taa.value,
                threshold=float(args.threshold_epitope),
                threshold_distance=float(args.threshold_distance),
                antigen_const=instance.t_var.value,
                hla_const=instance.t_allele.value,
                overlap_const=model.overlap,
                distance2self=int(args.distance is not None),
                uncertainty=int(args.uncertainty is not None),
                immunogenicity=float("{0:.2f}".format(-obj1)),
                risk=obj2,
                covered_hlas=float("{0:.2f}".format(float(len(cov_als))/float(len(instance.A)))),
                covered_antigens=float("{0:.2f}".format(float(len(cov_anti))/float(len(instance.Q))))
            ))
            f.write("\n#ES-{number}\n".format(number=i))
            header = "neoepitope\ttype\tgenes\tmutations\tHLAs"
            if args.distance:
                header+="\tdistance"
            header+="\t"+"\t".join(a.name+"_imm" for a in hlas)+"\t"+"\t".join(a.name+"_dist" for a in hlas)
            f.write(header+"\n")
            proteins = []
            for e in es:
                seq = str(e)
                proteins.extend([p.gene_id for p in e.get_all_proteins()])
                f.write("{seq}\t{type}\t{genes}\t{mutations}\t{alleles}\t{predict}\t{distance}\n".format(
                    seq=seq,
                    type="TSA" if e.get_metadata("taa", only_first=True) is None else e.get_metadata("taa",
                                                                                                      only_first=True),
                    genes=",".join(p.gene_id for p in e.get_all_proteins()),
                    mutations=",".join(set(pep_to_mutation[seq])),
                    alleles=",".join(str(a) for a in instance.A if seq in instance.A_I[a]),
                    distance="\t".join(str(distance.get((seq, a.name), 1.0)) for a in hlas),
                    predict="\t".join(str(float(df_epitope.loc[(e,"custom"),a])) for a in hlas)
                ))

            # protein section
            f.write("\n#PS-{number}\n".format(number=i))

            f.write("gene\tlog2(tumor_expression)\t"+"\t".join(tis)+"\n")
            for p in set(proteins):
                f.write("{gene}\t{tumor}\t{expression}\n".format(
                    gene=p,
                    tumor=float("{0:.2f}".format(math.log(expression[p],2) if expression[p] > 0.0 else 0.0)),
                    expression="\t".join(tissues.loc[(p,t),"Abundance"].iloc[0] if p in tissues.index.levels[0] else "Na" for t in tis)
                ))

            # allele section
            f.write("\n#AS-{number}\n".format(number=i))
            f.write("HLA\tlog2(tumor_expression)\n")

            hla_set = []
            for a in hlas:
            	if a.name not in hla_set:
                	f.write("{allele}\t{expr}\n".format(
                    	allele=a.name,
                    	expr=float("{0:.2f}".format(math.log(a.get_metadata("abundance",only_first=True),2)))
                	))
                	hla_set.append(a.name)
            f.write("\n\n\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
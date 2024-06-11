#!/usr/bin/env python

import itertools
import sys

import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

sys.path.append("..")
from utils import count


class Data:
    def __init__(self, protein_count):
        self.arr = np.empty((protein_count, 11201), dtype=np.float64)
        self.class_arr = np.empty(protein_count, dtype=int)
        self.group_arr = np.empty(protein_count, dtype=int)
        self.id_arr = np.empty(protein_count, dtype=int)

        aa = "AILMVNQSTGPCHKRDEFWY"
        sc = "11111222233455566777"
        self.sc_translator = aa.maketrans(aa, sc)

        AA = sorted([x for x in aa])
        SC = ["1", "2", "3", "4", "5", "6", "7"]

        self.di_pep = ["".join(i) for i in itertools.product(AA, repeat=2)]
        self.tri_pep = ["".join(i) for i in itertools.product(AA, repeat=3)]
        self.di_sc = ["".join(i) for i in itertools.product(SC, repeat=2)]
        self.tri_sc = ["".join(i) for i in itertools.product(SC, repeat=3)]
        self.tetra_sc = ["".join(i) for i in itertools.product(SC, repeat=4)]

    def feature_extract(self, raw_sequence):
        sequence = (
            raw_sequence.upper()
            .replace("X", "A")
            .replace("J", "L")
            .replace("*", "A")
            .replace("Z", "E")
            .replace("B", "D")
        )
        len_seq = len(sequence)
        sequence_sc = sequence.translate(self.sc_translator)

        di_pep_count = count.calculate_frequencies(sequence, self.di_pep, len_seq - 1)
        di_pep_count_n = np.asarray(di_pep_count, dtype=np.float64)

        tri_pep_count = count.calculate_frequencies(sequence, self.tri_pep, len_seq - 2)
        tri_pep_count_n = np.asarray(tri_pep_count, dtype=np.float64)

        di_sc_count = count.calculate_frequencies(sequence, self.di_sc, len_seq - 1)
        di_sc_count_n = np.asarray(di_sc_count, dtype=np.float64)

        tri_sc_count = count.calculate_frequencies(
            sequence_sc, self.tri_sc, len_seq - 2
        )
        tri_sc_count_n = np.asarray(tri_sc_count, dtype=np.float64)

        tetra_sc_count = count.calculate_frequencies(
            sequence_sc, self.tetra_sc, len_seq - 3
        )
        tetra_sc_count_n = np.asarray(tetra_sc_count, dtype=np.float64)

        X = ProteinAnalysis(sequence)
        biochemical_features = [
            X.isoelectric_point(),
            X.instability_index(),
            len_seq,
            X.aromaticity(),
            X.molar_extinction_coefficient()[0],
            X.molar_extinction_coefficient()[1],
            X.gravy(),
            X.molecular_weight(),
        ]

        biochemical_features_array = np.asarray(biochemical_features, dtype=np.float64)

        additional_biochemical_features = []
        additional_biochemical_features.append(X.secondary_structure_fraction()[0])
        additional_biochemical_features.append(X.secondary_structure_fraction()[1])
        additional_biochemical_features.append(X.secondary_structure_fraction()[2])
        additional_biochemical_features.append(
            (sequence.count("R") + sequence.count("L") + sequence.count("K")) / len_seq
        )  # acidic fraction
        additional_biochemical_features.append(
            (sequence.count("D") + sequence.count("E")) / len_seq
        )  # basic fraction
        additional_biochemical_features.append(
            np.mean(X.flexibility())
        )  # mean flexibility
        flex_smoothed = [
            np.mean(X.flexibility()[i : i + 5]) for i in range(len(X.flexibility()) - 5)
        ]
        additional_biochemical_features.append(
            max(flex_smoothed)
        )  # peak of smoothed flexibility
        additional_biochemical_features.append(
            X.flexibility().index(max(X.flexibility())) / len(X.flexibility())
        )  # peak flexibility relative position

        five_percent_range = int(round(len(X.flexibility()) * 0.05, 0))
        start_flexibility_mean = np.mean(X.flexibility()[:five_percent_range])
        end_flexibility_mean = np.mean(X.flexibility()[-five_percent_range:])
        additional_biochemical_features.append(
            (start_flexibility_mean + end_flexibility_mean) / 2
        )

        quintile_size = int(round(len(sequence) / 5, 0))
        seq_quintiles = [
            sequence[i * quintile_size : (i * quintile_size) + quintile_size]
            for i in range(5)
        ] + [sequence[quintile_size * 4 :]]

        for seq_quintile in seq_quintiles:
            print(seq_quintile)
            Xn = ProteinAnalysis(seq_quintile)
            additional_biochemical_features.append(Xn.isoelectric_point())
            additional_biochemical_features.append(Xn.gravy())
            additional_biochemical_features.append(
                (
                    seq_quintile.count("R")
                    + seq_quintile.count("L")
                    + seq_quintile.count("K")
                )
                / len(seq_quintile)
            )  # acidic fraction
            additional_biochemical_features.append(
                (seq_quintile.count("D") + seq_quintile.count("E")) / len(seq_quintile)
            )  # basic fraction
            additional_biochemical_features.append(np.mean(Xn.flexibility()))

        additional_biochemical_features_array = np.asarray(
            additional_biochemical_features, dtype=np.float64
        )
        print(additional_biochemical_features)

        row = np.concatenate(
            (
                di_pep_count_n,
                tri_pep_count_n,
                di_sc_count_n,
                tri_sc_count_n,
                tetra_sc_count_n,
                biochemical_features_array,
                additional_biochemical_features_array,
            )
        )
        row = row.reshape((1, row.shape[0]))

        return row

    def add_to_array(self, row, row_num, cls_number, group):
        self.arr[row_num, :] = row
        self.class_arr[row_num] = cls_number
        self.group_arr[row_num] = group
        self.id_arr[row_num] = row_num


def fasta_count(file_list):
    total_seqs = 0
    for file in file_list:
        for _ in SeqIO.parse(file, "fasta"):
            total_seqs += 1
    return total_seqs

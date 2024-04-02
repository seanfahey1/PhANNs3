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
        additional_features = [
            X.isoelectric_point(),
            X.instability_index(),
            len_seq,
            X.aromaticity(),
            X.molar_extinction_coefficient()[0],
            X.molar_extinction_coefficient()[1],
            X.gravy(),
            X.molecular_weight(),
        ]

        additional_features_array = np.asarray(additional_features, dtype=np.float64)

        row = np.concatenate(
            (
                di_pep_count_n,
                tri_pep_count_n,
                di_sc_count_n,
                tri_sc_count_n,
                tetra_sc_count_n,
                additional_features_array,
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

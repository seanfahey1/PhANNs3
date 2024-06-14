def calculate_frequencies(sequence: str, k_mers: list, sequence_len: int):
    count_list = []

    for k_mer in k_mers:
        count_list.append(sequence.count(k_mer))
    count_list = [x / len(sequence) for x in count_list]

    return count_list

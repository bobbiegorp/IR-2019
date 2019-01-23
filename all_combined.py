#!/usr/bin/env python3

import generate_input
import interleaving as il
import click_model_v2 as cm
import power_analysis as pa


def main():

    length_interleaving = 3

    inputs = generate_input.gen_input_pairs(length_interleaving, 2)

    #Click model training
    print('LOG :: TRAINING')
    rcm = cm.RCM()
    pbm = cm.PBM()
    database = cm.read_yandex("./YandexRelPredChallenge.txt")
    rcm.learn(database, length_interleaving)
    pbm.learn(database, 3, 5, length_interleaving)
    print('LOG :: DONE TRAINING')

    n_simulations = 500
    click_model_fs = [rcm.get_clicks, pbm.get_clicks]
    interleaving_fs = [il.td_interleaving, il.prob_interleaving]
    bin_set_labels = [
        'RCM & Team-Draft Interleaving',
        'RCM & Probabilistic Interleaving',
        'PBM & Team-Draft Interleaving',
        'PBM & Probabilistic Interleaving'
    ]

    n_bins = 10
    cut_sides = 0.05
    bins = [[[] for _ in range(n_bins)]
        for _ in range(len(bin_set_labels))]

    for l, pair in enumerate(inputs):

        print('LOG :: ' + str(l + 1) + ' / ' + str(len(inputs))
            + ' INPUTS')

        dERR = generate_input.ERR(pair[1]) - generate_input.ERR(pair[0])
        if dERR >= cut_sides and dERR < 1.0 - cut_sides:

            for i, click_model_f in enumerate(click_model_fs):
                for j, interleaving_f in enumerate(interleaving_fs):

                    ij = i*len(interleaving_fs)+j
                    print('  LOG :: ' + str(ij + 1) + ' / '
                        + str(len(bins)) + ' BINS')

                    permutations = generate_input.add_conflicts(pair)

                    for k, permutation in enumerate(permutations):

                        print('    LOG :: ' + str(k + 1) + ' / '
                            + str(len(permutations)) + ' PERMUTATIONS')

                        p = pa.interleaving_simulation(
                            permutation, n_simulations,
                            interleaving_f, click_model_f,
                            length_interleaving)
                        bins[ij][int(dERR * 10)].append(
                            pa.compute_sample_size(p))

    bin_sets = []
    bin_labels = pa.get_bin_labels(n_bins)
    for bin_set, label in zip(bins, bin_set_labels):
        print('===== ' + label + ' =====')
        bin_info = pa.process_bins(bin_set)
        pa.print_bin_info(bin_info)
        bin_sets.append(bin_info)

    pa.plot_bin_info(bin_sets, bin_set_labels, bin_labels)

    return

if __name__ == '__main__':
    main()



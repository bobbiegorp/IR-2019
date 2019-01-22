
import generate_input
import interleaving
import click_model_v2 as cm


def main():
    inputs = generate_input.gen_input_pairs(3, 2)
    #Click model train

    bins_prob_random = [[] for _ in range(10)]
    bins_prob_pbm = [[] for _ in range(10)]
    bins_td_random = [[] for _ in range(10)]
    bins_td_pbm = [[] for _ in range(10)]
    bins = [bins_prob_random, bins_prob_pbm, bins_td_random, bins_td_pbm]
    length_interleaving = 3

    rcm = cm.RCM()
    pbm = cm.PBM()
    database = cm.read_yandex("./YandexRelPredChallenge.txt")
    rcm.learn(database)
    pbm.learn(database,3,5,length_interleaving)

    click_model_functions = [rcm.get_clicks,pbm.get_clicks]
    interleaving_functions = [interleaving.td_interleaving, interleaving.prob_interleaving]

    for pair in inputs:
        dERR = generate_input.ERR(pair[1]) - generate_input.ERR(pair[0])
        if dERR >= 0.05 and dERR <= 0.95:

            for permutation in generate_input.add_conflicts(pair): # Assumes (non-) duplicates for now.

                for i,click_model in enumerate(click_model_functions):
                    for j,interleaving in enumerate(interleaving_functions):
                        p = interleaving_simulation(
                            permutation, 100, interleaving,click_model)
                        bins[i*len(interleaving_functions)+j][int(dERR * 10)].append(compute_sample_size(p))

    process_bins(bins)
    return


if __name__ == '__main__':
    main()




import generate_input
import interleaving as il
import click_model_v2 as cm
import power_analysis as pa

def main():

    inputs = generate_input.gen_input_pairs(3, 2)
    #Click model train

    bins_prob_random = [[] for _ in range(10)]
    bins_prob_pbm = [[] for _ in range(10)]
    bins_td_random = [[] for _ in range(10)]
    bins_td_pbm = [[] for _ in range(10)]
    bins = [bins_prob_pbm, bins_td_pbm, bins_prob_random,bins_td_random]
    length_interleaving = 3

    rcm = cm.RCM()
    pbm = cm.PBM()
    database = cm.read_yandex("./YandexRelPredChallenge.txt")
    rcm.learn(database)
    pbm.learn(database,3,5,length_interleaving)

    click_model_functions = [pbm.get_clicks,rcm.get_clicks]
    interleaving_functions = [il.prob_interleaving,il.td_interleaving]

    counter = 0
    for pair in inputs:
        dERR = generate_input.ERR(pair[1]) - generate_input.ERR(pair[0])
        if dERR >= 0.05 and dERR <= 0.95:

            for permutation in generate_input.add_conflicts(pair): # Assumes (non-) duplicates for now.

                for i,click_model_f in enumerate(click_model_functions):
                    for j,interleaving_f in enumerate(interleaving_functions):
                        p = pa.interleaving_simulation(
                            permutation, 100, interleaving_f,click_model_f,length_interleaving)
                        counter+=1
                        bins[i*len(interleaving_functions)+j][int(dERR * 10)].append(pa.compute_sample_size(p))

    print("Counter ", counter)
    for bins_table in bins:
        print("----------------------------------Table-----------------------------------\n")
        pa.process_bins(bins_table)
    return

if __name__ == '__main__':
    main()



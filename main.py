import keras
from deCNN import deCNN
import numpy as np
import time
import keras.backend
import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ######## Algorithm parameters ##################
    
    # dataset = "mnist"
    # dataset = "mnist-rotated-digits"
    # dataset = "mnist-rotated-with-background"
    # dataset = "rectangles"
    # dataset = "rectangles-images"
    dataset = "convex"
    # dataset = "mnist-random-background"
    # dataset = "mnist-background-images"
    
    number_runs = 20
    number_iterations = 20
    population_size = 20

    batch_size = 200
    batch_size_full_training = 200
    
    epochs_de = 1
    epochs_full_training = 100
    
    max_conv_output_channels = 256
    max_fully_connected_neurons = 300

    min_layer = 3
    max_layer = 10

    # Probability of each layer type (should sum to 1)
    probability_convolution = 0.6
    probability_pooling = 0.3
    probability_fully_connected = 0.1

    max_conv_kernel_size = 7

    Cg = 0.5
    dropout = 0.5

    ########### Run the algorithm ######################
    results_path = "./results/" + dataset + "/"

    if not os.path.exists(results_path):
            os.makedirs(results_path)

    all_best_metrics = np.zeros((number_runs, 2))
    runs_time = []
    all_best_par = []
    best_best_acc = 0

    for i in range(number_runs):
        print("Run number: " + str(i))
        start_time = time.time()
        de = deCNN(dataset=dataset, n_iter=number_iterations, pop_size=population_size,
                     batch_size=batch_size_de, epochs=epochs_de, min_layer=min_layer, max_layer=max_layer,
                     conv_prob=probability_convolution, pool_prob=probability_pooling,
                     fc_prob=probability_fully_connected, max_conv_kernel=max_conv_kernel_size,
                     max_out_ch=max_conv_output_channels, max_fc_neurons=max_fully_connected_neurons,
                     dropout_rate=dropout)

        de.fit(Cg=Cg, dropout_rate=dropout)

        print(de.best_acc)

        # Plot current best
        matplotlib.use('Agg')
        plt.plot(de.best_acc)
        plt.xlabel("Iteration")
        plt.ylabel("best acc")
        plt.savefig(results_path + "iter-" + str(i) + ".png")
        plt.close()


        end_time = time.time()

        running_time = end_time - start_time

        runs_time.append(running_time)

        # Fully train the best model found
        n_parameters = de.fit_best(batch_size=batch_size_full_training, epochs=epochs_full_training, dropout_rate=dropout)
        all_best_par.append(n_parameters)

        # Evaluate the fully trained best model
        best_metrics = de.evaluate_best(batch_size=batch_size_full_training)

        if best_metrics[1] >= best_best_acc:
            best_best_acc = best_metrics[1]

            # Save best best model
            best_best_yaml = de.best.model.to_yaml()

            with open(results_path + "best-best-model.yaml", "w") as yaml_file:
                yaml_file.write(best_best_yaml)
            
            # Save best best model weights to HDF5 file
            de.best.model.save_weights(results_path + "best-best-weights.h5")

        all_best_metrics[i, 0] = best_metrics[0]
        all_best_metrics[i, 1] = best_metrics[1]

        print("This run took: " + str(running_time) + " seconds.")

         # Compute mean accuracy of all runs
        all_best_mean_metrics = np.mean(all_best_metrics, axis=0)

        np.save(results_path + "/time_to_run.npy", runs_time)

        # Save all best metrics
        np.save(results_path + "/all_best_metrics.npy", all_best_metrics)

        # Save results in a text file
        output_str = "All best number of parameters: " + str(all_best_par) + "\n"
        output_str = output_str + "All best test accuracies: " + str(all_best_metrics[:,1]) + "\n"
        output_str = output_str + "All running times: " + str(runs_time) + "\n"
        output_str = output_str + "Mean loss of all runs: " + str(all_best_mean_metrics[0]) + "\n"
        output_str = output_str + "Mean accuracy of all runs: " + str(all_best_mean_metrics[1]) + "\n"

        print(output_str)

        with open(results_path + "/final_results.txt", "w") as f:
            try:
                print(output_str, file=f)
            except SyntaxError:
                print >> f, output_str

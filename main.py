import Visualisation
import preprocessing_data as ppd

import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import NNModel
from NNWrapper import NNWrapper
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time
import ApplicationSetup as AS
import CONSTANTS as c
# import DataWrapperNN as DWNN
import CustomLosses as losses
import ModelParameters as MP
import Visualisation as V
import utils as u
dev = "cpu"  # AS.if_cuda(); TODO: should be fixed
AS.application_setup()

tasks = [c.Label, c.ctDNADetected, c.VAFg0p001]
task = c.ctDNADetected
def main():
    input_file = "/Users/alexandra/PhD/FragmentationPatterns/Data/presentation_input2/window_fragsize_short_noReads_byArmNoScale.csv"
    test_studies = ["PearlStudy"]
    validation_studies = ["NeoRheaStudy", "healthy_sWGS"]
    test_data, validation_data, test_labels, validation_labels \
        = ppd.get_data_for_wrapper(input_file, test_studies, validation_studies, task)

    # V.plot_input(input_file)

    losses_global = [losses.MSE()]
    MainModel = NNModel.NeuralNetworkModel(input_dim_amino_acid=1,
                                           input_dim_chain=671, output_dim=39,
                                           hidden_layers=30, dev=dev, batch_size=MP.batch_size)

    # dataset = np.array(data)
    # kf = KFold(n_splits=MP.cv)  # rmsd = []    mse = []
    nr_epochs = np.arange(1, 30, 1)
    x_train, x_test = train_test_split(test_data, test_size=0.3, random_state=123, stratify=test_labels[:, task])

    wrapper = NNWrapper(MainModel, losses_global)  # builds the wrapper of the NN
    mean_mse_train = []
    mean_mse_test = []
    mse_train = []
    mse_test = []
    for epoch in nr_epochs:
        cv_start = time.time()

        start = time.time()
        wrapper.fit(x_train, epochs=epoch, lr_scheduler_reduce=MP.lr_scheduler_reduce,
                    batch_size=MP.batch_size, save_model_every=MP.save_model_sec, weight_decay=MP.weight_decay,
                    learning_rate=MP.learning_rate, LOG=MP.LOG, learn_sum=MP.learn_sum)
        end = time.time()
        # print("### time to train: ", end - start, " for ", epoch, "epochs")
        # prediction on train dataset
        ypred_train, loss_dataset_train, s_pd_train, s_rd_train, patientId_train, ctDNADetection_train,  accuracy_train \
            = wrapper.predict(x_train, batch_size=MP.batch_size)
        # our actual prediction
        start = time.time()
        ypred_test, loss_dataset_test, s_pd_test, s_rd_test, patientId_test, ctDNADetection_test, accuracy_test = \
            wrapper.predict(x_test, batch_size=MP.batch_size)
        end = time.time()
        # print("### time to predict: ", end - start, " for ", epoch, "epochs")

        mse_train += loss_dataset_train
        mse_test += loss_dataset_test
        patientId_test += patientId_test

        mean_mse_train += [np.mean(mse_train)]
        mean_mse_test += [np.mean(mse_test)]
        cv_end = time.time()
        if epoch % 10 == 0:
            print("### epoch ", epoch, "\nLoss on Train:", mean_mse_train, "\nLoss on Test", mean_mse_test)
            # print("### time to train and predict: ", cv_end - cv_start, " for ", epoch, "epochs")
            # print("ypred_test", ypred_test)
            accuracy_train_mean = np.mean(accuracy_train)
            accuracy_test_mean = np.mean(accuracy_test)
            print("### epoch ", epoch, "\nAccuracy on Train:", accuracy_train_mean, "\nAccuracy on Test", accuracy_test_mean)
            print("ypred_test", ypred_test)
            print("patientId_test", patientId_test)
            print("model predictions: ", s_pd_test)
            print("truth: ", s_rd_test)
        # print(f"Epoch: {epoch} | Loss: {mean_mse_train:.5f}, Accuracy: {accuracy_train_mean:.2f}% | Test loss: {mean_mse_test:.5f}, Test acc: {accuracy_test_mean:.2f}%")
    mean_mse_train = [np.mean(mse_train)]
    mean_mse_test = [np.mean(mse_test)]
    print("Mean loss function on Train:", mean_mse_train, "\n Mean loss on Test", mean_mse_test)

    # for epoch in nr_epochs:
    # dataset[91:92]
    # dataset_range = np.arange(0, 179, 1)
    # for i in dataset_range:
    # wrapper.fit(dataset[156:156+1], epochs=500, lr_scheduler_reduce=MP.lr_scheduler_reduce,
    #             batch_size=MP.batch_size, save_model_every=MP.save_model_sec, weight_decay=MP.weight_decay,
    #             learning_rate=MP.learning_rate, LOG=MP.LOG, learn_sum=MP.learn_sum)
    #
    # # prediction on train dataset
    # ypred_train, all_residue_train, true_torsion_angles_train, kf_mse_train, predicted_coordinates_train, \
    # true_coordinates_train, atom_labels_train, peptide_names \
    #     = wrapper.predict(dataset[156:156+1], batch_size=MP.batch_size, epochs= 500)
    # # our actual prediction
    # ypred_test, all_residue_test, true_torsion_angles_test, kf_mse_test, predicted_coordinates_test, \
    # true_coordinates_test, atom_labels, peptide_names_test \
    #     = wrapper.predict(dataset[156:156+1], batch_size=MP.batch_size,epochs= 500)
    # dv.plot_histo_RMSD(peptide_names_test, kf_mse_test, 500)
    # dv.plot_histo_unif_distrib(MainModel.alphabet)
    # # mean_mse_train += [np.mean(kf_mse_train)]
    # # mean_mse_test += [np.mean(kf_mse_test)]
    # print("peptide name: ", peptide_names_test, "\n")
    # print("### epoch ", 500, "\nMSE on Train:", np.mean(kf_mse_train), "\nMSE on Test", np.mean(kf_mse_test))
    # 2

    ## final version with kfold validation
    # for epoch in nr_epochs:
    #     # train the model
    #     mse_train = []
    #     mse_test = []
    #     peptide_names = []
    #     ypred_test = []
    #     true_torsion_angles_test = []
    #     predicted_coordinates_test = []
    #     true_coordinates_test = []
    #     atom_labels_train = []
    #     atom_labels = []
    #     all_residue_test = []
    #     cv_start = time.time()
    #     for train, test in kf.split(dataset):
    #         trainStructures = dataset[train]  # training structures
    #         testStructures = dataset[test]  # testing structures
    #         start = time.time()
    #         wrapper.fit(trainStructures, epochs=epoch, lr_scheduler_reduce=MP.lr_scheduler_reduce,
    #                     batch_size=MP.batch_size, save_model_every=MP.save_model_sec, weight_decay=MP.weight_decay,
    #                     learning_rate=MP.learning_rate, LOG=MP.LOG, learn_sum=MP.learn_sum)
    #         end = time.time()
    #         print("### time to train: ", end - start, " for ", epoch, "epochs")
    #         # prediction on train dataset
    #         ypred_train, all_residue_train, true_torsion_angles_train, kf_mse_train, predicted_coordinates_train, \
    #         true_coordinates_train, atom_labels_train, peptide_names_kf_train \
    #             = wrapper.predict(trainStructures, batch_size=MP.batch_size)
    #         # our actual prediction
    #         start = time.time()
    #         ypred_test, all_residue_test, true_torsion_angles_test, kf_mse_test, predicted_coordinates_test, \
    #         true_coordinates_test, atom_labels, peptide_names_kf_test \
    #             = wrapper.predict(testStructures, batch_size=MP.batch_size)
    #         end = time.time()
    #         print("### time to predict: ", end - start, " for ", epoch, "epochs")
    #         mse_train += kf_mse_train
    #         mse_test += kf_mse_test
    #         peptide_names += peptide_names_kf_test
    #         #
    #     # dv.plot_histo_RMSD(peptide_names, mse_test, epoch)
    #     mean_mse_train += [np.mean(mse_train)]
    #     mean_mse_test += [np.mean(mse_test)]
    #     cv_end = time.time()
    #
    #     print("### epoch ", epoch, "\nMSE on Train:", np.mean(mse_train), "\nMSE on Test", np.mean(mse_test))
    #     print("### time to train and predict: ", cv_end - cv_start, " for ", epoch, "epochs")
    #     print("ypred_test", ypred_test)
    #     print("true angles", true_torsion_angles_test)
    #     print("predicted_coordinates", predicted_coordinates_test)
    #     print("true_coordinates", true_coordinates_test)
    #     print("atom_labels_train", atom_labels_train)
    #     print("atom_labels_pred", atom_labels)
    #     print("all_residue_test", all_residue_test)

    print('The monkeys are listening')
    # print("FINAL CV")
    # dv.draw_result(nr_epochs, mean_mse_train, mean_mse_test, "RMDS value over the epochs")

    # print("Embeddings: ", MainModel.embeds(torch.arange(0, 20)))
    # dv.draw_aminoacids_emb_pca(MainModel.embeds(torch.arange(0, 20)))


if __name__ == '__main__':
    import sys

    sys.exit(main())

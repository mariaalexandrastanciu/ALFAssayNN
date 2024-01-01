# Created by alexandra at 21/08/2023

from torch.utils.data import DataLoader
import DataWrapperNN as dwnn
import ModelParameters as MP
import time
import utils as u
import numpy as np
import torch as t
from Logger import MetaLogger
from sklearn import preprocessing
from helper_functions import plot_predictions, plot_decision_boundary
import matplotlib.pyplot as plt


class NNWrapper:
    def __init__(self, MainModel, losses):
        self.model = MainModel
        self.losses = losses
        self.dev = MainModel.dev

    def predict(self, input_data, batch_size):
        self.model.eval()  # removes the noise inserted by dropouts

        # duplicates random input data such that batch size to be multiple of sample size
        input_data = dwnn.batch_padding(input_data, batch_size)
        loader = dwnn.dataWrapper(batch_size=batch_size, num_workers=0, data=input_data, shuffle=False)


        pred_ys = []
        loss_dataset = []
        start_time = time.time()
        # sample real data
        s_rd = []
        # sample predicted data
        s_pd = []
        patientIds = []
        ctDNADetectionList = []
        accuracies = []
        for si, sample in enumerate(loader):
            # true_fragment_lengths, true_y = sample
            # true_fragment_lengths, sampleNames, true_y, ctDNADetection, ctDNAbyVAF = sample
            fragmentSize, sampleNames, label, ctDNADetection, ctDNAbyVAF = sample
            predicted_arm_labels = self.model(fragmentSize)
            # true_label_per_arm = label.repeat(39)
            true_ctDNADetection_per_arm = ctDNADetection.view(batch_size, 1).repeat(1, 39) #ctDNADetection.repeat(39)
            # predicted_label = t.mean(predicted_arm_labels).view(1)

            loss = t.nn.CrossEntropyLoss()
            # loss_tensor = loss(true_label_per_arm, predicted_arm_labels) + eps
            loss_tensor = loss(true_ctDNADetection_per_arm, predicted_arm_labels)

            # plt.figure(figsize=(12, 6))
            # plt.subplot(1, 2, 1)
            # plt.title("Train")
            # plot_decision_boundary(self.model, fragmentSize, true_ctDNADetection_per_arm)
            # plt.subplot(1, 2, 2)
            # plt.title("Test")
            # plot_decision_boundary(model_0, X_test, y_test)

            pred_ys += [t.round(t.mean(predicted_arm_labels))]
            patientIds += sampleNames
            ctDNADetectionList += ctDNADetection

            # TODO: add the function which transforms fragment lengths to VAF; for now we predict only the frag
            #  lenght counts

            # backbone_pred_coords, sidechain_pred_coords, all_predicted_coordinates = \
            #     u.get_coodintes(predicted_angles, batch_size, residues, true_y[dw.i_peptide_atom_names])
            # backbone_pred_coords, backbone_atom_labels = u.get_backbone_coodintes(predicted_angles, batch_size, residues,
            #                                                 true_y[dw.i_peptide_atom_names])

            # loss function
            # all coordinates
            # _, loss_value = self.losses[0](true_y[dwnn.i_peptide_coordinates], all_predicted_coordinates)
            # backbone
            # loss_tensor, loss_value = self.losses[1](true_y[dw.i_peptide_backbone_coordinates],
            #                                          backbone_pred_coords)
            # MSE angles
            # le = preprocessing.LabelEncoder()
            # le.fit(true_y[dwnn.Label])
            # y_true_int = le.transform(true_y[dwnn.Label])
            # _, loss_value = self.losses[0](y_true_int, labels)

            s_rd += true_ctDNADetection_per_arm  # true_coordinates
            s_pd += predicted_arm_labels
            r_predicted_arm_labels = t.round(predicted_arm_labels)
            accuracies += [u.accuracy_fn(true_ctDNADetection_per_arm, r_predicted_arm_labels)]
            # if true_y[dw.i_protein_name][0][0] != "3dgj_2" and true_y[dw.i_protein_name][0][0] != "2onw" and true_y[
            #     dw.i_protein_name][0][0] != "3ow9" and true_y[dw.i_protein_name][0][0] != "4ubz" and true_y[
            #     dw.i_protein_name][0][0] != "5wkd_b_2":
            #     pass
            # # dv.create_pdb_file(residues, true_y[dw.i_peptide_atom_names], all_predicted_coordinates,
            # #                    true_y[dw.i_protein_name], epochs)
            # else:
            #     dv.create_pdb_file(residues, true_y[dw.i_peptide_atom_names], all_predicted_coordinates,
            #                        true_y[dw.i_protein_name], epochs)

            loss_dataset += [float(loss_tensor.data.cpu().numpy())]

        print("\tprediction Time %s s" % round((time.time() - start_time), 2))
        return pred_ys, loss_dataset, s_pd, s_rd, patientIds, ctDNADetectionList,  accuracies

    def fit(self, input_data, epochs=5, lr_scheduler_reduce=0.5, batch_size=1, save_model_every=10,
            weight_decay=1e-2, learning_rate=1e-3, LOG=True, learn_sum=False):

        if LOG:
            self.logger = MetaLogger(self.model, port=6001)

        ### data loader ###
        input_data = dwnn.batch_padding(input_data, batch_size)
        loader = dwnn.dataWrapper(batch_size=batch_size, num_workers=0, data=input_data)

        #######PRINT INFO##############
        parameters = list(self.model.parameters())
        used_params = []
        for i in parameters:
            if i.requires_grad:
                used_params += list(i.data.cpu().numpy().flat)
        # print('\tNumber of parameters=', len(used_params))

        ########OPTIMIZER##########
        optimizer = t.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_scheduler_reduce,
                                                           patience=10000, verbose=True, threshold=1e-4,
                                                           threshold_mode='rel', cooldown=0, min_lr=1e-10, eps=1e-08)
        epoch_iteration = 0
        loss_value = 0.0
        t.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        all_losses = []
        all_pears_cor = []
        lrs = []
        while epoch_iteration < epochs:
            start = time.time()
            epoch_losses = []
            pears_cor = []
            for sample_iteration, sample in enumerate(loader):
                # see doc on true_y in dw.multiTaskCollate() function
                fragmentSize, sampleNames,  label, ctDNADetection, ctDNAbyVAF = sample
                predicted_arm_labels = self.model(fragmentSize)  # calls forward
                # true_label_per_arm = label.repeat(39)
                # predicted_arm_labels = t.round(predicted_arm_labels).squeeze()
                true_ctDNADetection_per_arm = ctDNADetection.view(batch_size, 1).repeat(1, 39)
                # predicted_label = t.mean(predicted_arm_labels).view(1)
                # backbone_pred_coords,_ = u.get_backbone_coodintes(predicted_angles, batch_size, residues,
                #                                                 true_y[dw.i_peptide_atom_names])
                # backbone_pred_coords, sidechain_pred_coords, all_predicted_coordinates = \
                #     u.get_coodintes(predicted_angles, batch_size, residues, true_y[dw.i_peptide_atom_names])


                # all fragments MSE
                loss = t.nn.CrossEntropyLoss()
                # eps = 1e-6
                # loss_tensor = loss(true_label_per_arm, predicted_arm_labels) + eps
                loss_tensor = loss(true_ctDNADetection_per_arm, predicted_arm_labels) #+ eps
                ## regression
                # loss_tensor, loss_value = self.losses[0](labels[0][dwnn.Label], predicted_y)

                # backbone coordinates loss
                # loss_tensor, loss_value = self.losses[1](true_y[dw.i_peptide_backbone_coordinates],
                #                            backbone_pred_coords)

                # MSE
                #loss_tensor, loss_value = self.losses[0](true_y[dw.i_p_torsion_angles], predicted_angles)
                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:
                #         print(name, param.data)

                epoch_losses += [float(loss_tensor.data.cpu().numpy())]
                loss_tensor.backward()


                if LOG:
                    self.logger.update_weights(epoch_iteration)

                optimizer.step()
                optimizer.zero_grad()

                # hidden.detach_()
                # hidden = hidden.detach()

            end = time.time()
            if epoch_iteration % 10 == 0:
                print(np.mean(epoch_losses))
                print(" epoch ", epoch_iteration, "time", round(end - start, 2))

            scheduler.step(np.mean(epoch_losses))
            if LOG:
                self.logger.writeTensorboardLog(epoch_iteration, np.mean(epoch_losses), end - start,
                                                self.model.embeds(t.arange(0, 20)))

            epoch_iteration += 1
            all_losses += [np.mean(epoch_losses)]

        #self.logger.shutdown()

        def get_params(self, deep=False):
            return {}
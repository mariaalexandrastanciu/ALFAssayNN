# Created by alexandra at 18/12/2023
import numpy as np
import pandas as pd
import CONSTANTS as c

def create_data(file, metadata_file, output):
    data = pd.read_csv(file, sep="\t")
    metadata = pd.read_csv(metadata_file, sep="\t")

    melt_short_reads_columns = ["short_arm_" + str(x) for x in c.armlevels]
    # melt_short_reads_columns.extend(["PatientId"])

    melt_no_reads_columns = ["NoReads_arm_" + str(x) for x in c.armlevels]
    # melt_no_reads_columns.extend(["PatientId"])

    # data_short_reads = data[melt_short_reads_columns]
    data_short_reads_melted = pd.melt(data, id_vars=[("PatientId")], value_vars=[*melt_short_reads_columns],
                              var_name='region', value_name='short_reads')
    data_short_reads_melted["region"] = data_short_reads_melted["region"].str.replace("short_arm_", "")
    # data_no_reads = data[melt_no_reads_columns]
    data_no_reads_melted = pd.melt(data, id_vars=[("PatientId")],
                                      value_vars=[*melt_no_reads_columns],
                                      var_name='region', value_name='no_reads')
    data_no_reads_melted["region"] = data_no_reads_melted["region"].str.replace("NoReads_arm_", "")

    melted_data = pd.merge(data_short_reads_melted, data_no_reads_melted, on=["PatientId", "region"])
    melted_data = pd.merge(metadata, melted_data, on="PatientId")

    melted_data.to_csv(output, sep="\t", index=False)

#
# metadata_file = "/Users/alexandra/PhD/FragmentationPatterns/Data/MetaData/AllStudiesMetaData.csv"
# file = "/Users/alexandra/PhD/FragmentationPatterns/Data/version1/window_fragsize_short_noReadsByArmNoScale.csv"
# output = "/Users/alexandra/PhD/FragmentationPatterns/Data/presentation_input2/window_fragsize_short_noReads_byArmNoScale.csv"
# create_data(file, metadata_file, output)

def get_data_for_wrapper(file, test_studies, validation_studies, task ):
    data = pd.read_csv(file, sep="\t")
    data = data.dropna(subset=[c.task_dict.get(task)])

    data["Label"] = np.where(data["Label"] == "Cancer", 1, 0)
    data["ctDNADetected"] = np.where(data["ctDNADetected"] == True, 1, data["ctDNADetected"])
    data["ctDNADetected"] = np.where(data["ctDNADetected"] == False, 0, data["ctDNADetected"])
    data["ctDNADetected"] = np.where(data["ctDNADetected"].isin([0, 1]), data["ctDNADetected"], -1)

    data["VAFg0p001"] = np.where(data["VAFg0p001"] == True, 1, data["VAFg0p001"])
    data["VAFg0p001"] = np.where(data["VAFg0p001"] == False, 0, data["VAFg0p001"])
    data["VAFg0p001"] = np.where(data["VAFg0p001"].isin([0, 1]), data["VAFg0p001"], -1)

    test_data = data[data["study"].isin(test_studies)]
    validation_data = data[data["study"].isin(validation_studies)]
    features_columns = ["short_reads", "no_reads"]
    labels_columns = ["PatientId", "Label", "ctDNADetected", "VAFg0p001"]

    test_labels = test_data[labels_columns].drop_duplicates().values
    fragmentSizeSampleTest = []
    for i, sample in enumerate(test_labels):
        features_per_sample = test_data[test_data["PatientId"]==sample[0]][features_columns]
        test_labels_per_sample=test_labels[i]
        fragmentSizeSampleTest.append([features_per_sample, test_labels_per_sample])

    validation_labels = validation_data[labels_columns].drop_duplicates().values
    fragmentSizeSampleValidation = []
    for i, sample in enumerate(validation_labels):
        features_per_sample = validation_data[validation_data["PatientId"]==sample[0]][features_columns]
        validation_labels_per_sample=validation_labels[i]
        fragmentSizeSampleValidation.append([features_per_sample, validation_labels_per_sample])


    return fragmentSizeSampleTest, fragmentSizeSampleValidation, test_labels, validation_labels


# file = "/Users/alexandra/PhD/FragmentationPatterns/Data/presentation_input2/window_fragsize_short_noReads_byArm.csv"
# test_studies = ["PearlStudy"]
# validation_studies = ["NeoRheaStudy", "healthy_sWGS"]
# fragmentSizeSampleTest, fragmentSizeSampleValidation = get_data_for_wrapper(file, test_studies, validation_studies )
#

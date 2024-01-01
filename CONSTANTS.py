# Created by alexandra at 18/12/2023
chr_arms = ["1p", "1q", "2p", "2q", "3p", "3q", "4p", "4q", "5p", "5q", "6p", "6q",
            "7p", "7q", "8p", "8q", "9p", "9q", "10p", "10q", "11p", "11q", "12p",
            "12q", "13p", "13q", "14p", "14q", "15p", "15q", "16p", "16q", "17p", "17q", "18p", "18q",
             "19p", "19q", "20p", "20q", "21p", "21q", "22p", "22q"]

armlevels = ["1p","1q","2p","2q","3p","3q","4p","4q","5p","5q","6p","6q",
               "7p","7q","8p","8q", "9p", "9q","10p","10q","11p","11q","12p",
               "12q","13q","14q","15q","16p","16q","17p","17q","18p","18q",
               "19p", "19q","20p","20q","21q","22q"]

(PatientId, Label, ctDNADetected, VAFg0p001) = range(4)
task_dict = {PatientId: "PatientId", Label: "Label", ctDNADetected:"ctDNADetected", VAFg0p001:"VAFg0p001"}
import road_segmentation as rs

#if data_dir=None, assumes data is under default dir
data_dir_nic = "/media/nic/VolumeAcer/CIL_data"

rs.data.cil.preprocess_unsupervised_data(data_dir=data_dir_nic)


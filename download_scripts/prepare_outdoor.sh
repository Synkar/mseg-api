#!/bin/sh
MSEG_DST_DIR=$1
NUM_CORES_TO_USE=$2

mkdir -p $MSEG_DST_DIR/mseg_dataset
mkdir -p $MSEG_DST_DIR/mseg_dataset/BDD
mkdir -p $MSEG_DST_DIR/mseg_dataset/Cityscapes
mkdir -p $MSEG_DST_DIR/mseg_dataset/IDD
mkdir -p $MSEG_DST_DIR/mseg_dataset/KITTI
mkdir -p $MSEG_DST_DIR/mseg_dataset/MapillaryVistasPublic
mkdir -p $MSEG_DST_DIR/mseg_dataset/ScanNet
mkdir -p $MSEG_DST_DIR/mseg_dataset/WildDash

#CAMVID
./mseg_download_camvid.sh $MSEG_DST_DIR/mseg_dataset/Camvid
./mseg_remap_camvid.sh $MSEG_DST_DIR/mseg_dataset/Camvid $NUM_CORES_TO_USE

#BDD
./mseg_extract_bdd.sh $MSEG_DST_DIR/mseg_dataset/BDD 2>&1

#Cityscapes
./mseg_download_cityscapes.sh $MSEG_DST_DIR/mseg_dataset/Cityscapes 2>&1 "" "" #add credentials here
./mseg_extract_cityscapes.sh $MSEG_DST_DIR/mseg_dataset/Cityscapes 2>&1
./mseg_remap_cityscapes.sh $MSEG_DST_DIR/mseg_dataset/Cityscapes $NUM_CORES_TO_USE 2>&1

#IDD -- Remember to update download url
./mseg_download_idd.sh $MSEG_DST_DIR/mseg_dataset/IDD 2>&1
./mseg_remap_idd.sh $MSEG_DST_DIR/mseg_dataset/IDD $NUM_CORES_TO_USE 2>&1

#KITTI
./mseg_download_kitti.sh $MSEG_DST_DIR/mseg_dataset/KITTI 2>&1
./mseg_remap_kitti.sh $MSEG_DST_DIR/mseg_dataset/KITTI $NUM_CORES_TO_USE 2>&1

#Mapillary
./mseg_download_mapillary.sh $MSEG_DST_DIR/mseg_dataset/MapillaryVistasPublic 2>&1
./mseg_extract_mapillary.sh $MSEG_DST_DIR/mseg_dataset/MapillaryVistasPublic 2>&1
./mseg_remap_mapillary.sh $MSEG_DST_DIR/mseg_dataset/MapillaryVistasPublic $NUM_CORES_TO_USE 2>&1

#WildDash
./mseg_extract_wilddash.sh $MSEG_DST_DIR/mseg_dataset/WildDash 2>&1 
./mseg_remap_wilddash.sh $MSEG_DST_DIR/mseg_dataset/WildDash $NUM_CORES_TO_USE 2>&1

#Relabeling
./mseg_apply_relabeling.sh $NUM_CORES_TO_USE 2>&1
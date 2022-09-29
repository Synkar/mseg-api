import os
import json
import requests


def main(json_path, data_path):
	os.makedirs(data_path, exist_ok=True)
	imgs_path = os.path.join(data_path,'imgs')
	labels_path = os.path.join(data_path,'labels')
	metadata_path = os.path.join(data_path,'metadata')
	os.makedirs(imgs_path, exist_ok=True)
	os.makedirs(labels_path, exist_ok=True)
	os.makedirs(metadata_path, exist_ok=True)

	f = open(json_path)
	data = json.load(f)
	samples = data['dataset']['samples']
	i = 1
	for sample in samples:	
		img_name = sample['name']
		print(img_name, ' - ', i, '/', len(samples))
		
		img_url = sample['attributes']['image']['url']
		label_url = sample['labels']['ground-truth']['attributes']['segmentation_bitmap']['url']
		response = requests.get(img_url)
		open(os.path.join(imgs_path, img_name), "wb").write(response.content)
		response = requests.get(label_url)
		open(os.path.join(labels_path, img_name), "wb").write(response.content)
		#Save metadata as json
		json_object = json.dumps(sample, indent=4)
		with open(os.path.join(metadata_path, img_name.split('.')[0]+'.json'), "w") as outfile:
			outfile.write(json_object)
		i+=1

if __name__ == '__main__':
	json_path = '/home/hudson/Desktop/Synkar/Codes/semantic_segmentation/mseg-api/download_scripts/sidewalk-imagery-v1.0.json'
	dst_path = '/home/hudson/Desktop/Synkar/Codes/mseg/data/mseg_dataset/sidewalk_imagery'
	main(json_path, dst_path)

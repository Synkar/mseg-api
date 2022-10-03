from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset

# Initialize a SegmentsDataset from the release file
my_key = ''
client = SegmentsClient(my_key)
release = client.get_release('segments/sidewalk-imagery', 'v1.0') # Alternatively: release = 'flowers-v1.0.json'
dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

# Export to semantic format
export_dataset(dataset, export_format='semantic')
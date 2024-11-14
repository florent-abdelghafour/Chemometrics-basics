'''Credits: group 2 AgroTIC'''

import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA
from skimage.filters import threshold_multiotsu
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops


main_data_folder = './data/data_grp2'    

# Initialize the HSI dataset and define file extension: contains all paths of hdr and data files
dataset =HsiDataset(main_data_folder,data_ext='hyspex')
nb_images = len(dataset)
#check if data path exists !
if os.path.isdir(main_data_folder):
    if nb_images>0:
        print(f"dataset  is valid and contains {nb_images} image(s)")
    else:
        print('empty dataset')
else:
    print('path invalid')


# Initialize the HSI reader: class containg info about the image and allow to load and operate on the hsi
HSIreader = HsiReader(dataset)

#choose image index to read e.g. first image idx=0
#or loop trhough "for idx in range(len(dataset)):"
idx=0

HSIreader.read_image(idx) #reads without loading! to get metadata
metadata = HSIreader.current_metadata

#define wavelenrghts (for plots mostly)
wv =HSIreader.get_wavelength()
wv = [int(l) for l in wv]

# Get the hyperspectral data
hypercube= HSIreader.get_hsi()

#Sample some spectra to determine generic pca loadings
n_samples = 50000
x_idx = np.random.randint(0, hypercube.shape[0], size=n_samples)
y_idx = np.random.randint(0, hypercube.shape[1], size=n_samples)

spectral_samples = hypercube[x_idx, y_idx, :]

#PCA
nb_pca_comp =3
pca = PCA(n_components=nb_pca_comp)
pca_scores = pca.fit_transform(spectral_samples)
pca_loadings =pca.components_.T*np.sqrt(pca.explained_variance_)


default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']



#project back the loadings on the entire hsi to get scores
score_img = HSIreader.project_pca_scores(pca_loadings)



# automatic thresholding with Ostu method (histogram based)
score_pc_ref = score_img[:,:,1]   
thresholds = threshold_multiotsu(score_pc_ref, classes=3)
segmented = np.digitize(score_pc_ref, bins=thresholds)


#get a labelled image 
labeled_image = label(segmented)

#fill holes in small object
binary_image = labeled_image > 1
filled_binary_image = binary_fill_holes(binary_image)
    
# plt.figure()
# plt.imshow(filled_binary_image)
# plt.show(block=False)

#remove artefacts of segmentation
labeled_image = label(filled_binary_image)
labeled_image= label(remove_small_objects(labeled_image > 0, min_size=50))

# plt.figure()
# plt.imshow(labeled_image)
# plt.show(block=False)

color_image = color_labels(labeled_image)
    


min_kernel_size =  50
padding=50
horizontal_tolerance =50
regions = regionprops(labeled_image)
print(f"Number of regions found: {len(regions)}")
object_data = []

for region in regions:
    # Get the object ID (label)
    obj_id = region.label
    
    # Skip the background (label 0)
    if obj_id == 0:
        continue
    
    # Filter out regions based on pixel area
    if region.area < min_kernel_size:
        continue
    
    # Get the centroid coordinates
    centroid = region.centroid
    
    # Get the coordinates of all pixels belonging to this object
    pixel_coords = np.array(region.coords)  
    
    # Get the original bounding box of the region
    min_row, min_col, max_row, max_col = region.bbox
    
    # Expand the bounding box by padding, ensuring it stays within the image boundaries
    min_row = max(0, min_row - padding)
    min_col = max(0, min_col - padding)
    max_row = min(hypercube.shape[0], max_row + padding)
    max_col = min(hypercube.shape[1], max_col + padding)
    
    # Store in dictionary
    object_data.append({
        'id': obj_id,
        'centroid': centroid,
        'pixels': pixel_coords,
        'bbox': (min_row, min_col, max_row, max_col)  # Store the expanded bounding box
    })

object_data.sort(key=lambda obj: obj['centroid'][0])  # Sort by x-coordinate

columns_of_centroids = []
current_column = []

for obj in object_data:
    if not current_column:
        current_column.append(obj)
    else:
        last_centroid_x = current_column[-1]['centroid'][0]
        if abs(obj['centroid'][0] - last_centroid_x) < 5:
            current_column.append(obj)
        else:
            columns_of_centroids.append(current_column)
            current_column = [obj]


# Convert list to dictionary with ordered keys
for column in columns_of_centroids:
        column.sort(key=lambda obj: obj['centroid'][1])
        
# Initialize index counter
idx = 1
n_rows = max(len(column) for column in columns_of_centroids)  # Find max number of rows among columns

# Pad columns with None if they have fewer objects than the max row count
for column in columns_of_centroids:
    while len(column) < n_rows:
        column.append(None)

# Add grid_coord and idx to each object's dictionary
for row in range(n_rows):
    for col, column in enumerate(columns_of_centroids):
        obj = column[row]
        if obj:
            obj['grid_coord'] = (row + 1, col + 1)  # Store as (row, col) tuple
            obj['idx'] = idx
            idx += 1

# Sort the object_data list by 'idx' for final top-left to bottom-right order
object_data_sorted = sorted([obj for obj in object_data if 'idx' in obj], key=lambda obj: obj['idx'])


sauve = []
for i in range(len(object_data_sorted)):
    dico = object_data_sorted[i]["pixels"]
  
    didico = np.asarray(dico)

    grains = hypercube[didico[:,0], didico[:,1]] 

    sauve.append(grains) 
matrice = np.concatenate(sauve)

# Filter the part of the spectra if there are artefacts
bin_wv_threshold = 50
wv = wv[bin_wv_threshold:]
matrice = matrice[:,bin_wv_threshold:] 

nb_pca_comp =3
pca = PCA(n_components=nb_pca_comp)
pca_scores = pca.fit_transform(matrice)
pca_loadings =pca.components_.T*np.sqrt(pca.explained_variance_)
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(np.shape(pca_loadings)[1]):
    plt.figure()
    plt.plot(wv,pca_loadings[:,i],default_colors[i])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")  
    lab= 'PC'+str(i+1)
    plt.title(lab) 
    plt.grid()  
plt.show(block=False)

plt.figure()
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.6, c='blue', edgecolor='k', s=50)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Score Plot PC1/PC2")
plt.grid()
plt.show(block=False)


fig,(ax1,ax2,ax3)=plt.subplots(1,3)
ax1.grid()
ax1.scatter(pca_scores[:, 1], pca_scores[:, 2], alpha=0.6, c='blue', edgecolor='k', s=50)
ax1.set_xlabel("PC2")
ax1.set_ylabel("PC3")
ax1.set_title("Score Plot PC2/PC3")
ax2.grid()
ax2.scatter(pca_scores[:, 1], pca_scores[:, 2], alpha=0.6, c='blue', edgecolor='k', s=50)
ax2.set_xlabel("PC2")
ax2.set_ylabel("PC3")
ax2.set_ylim([-2000,1000])
ax2.set_title("Score Plot PC2/PC3")
ax3.grid()
ax3.scatter(pca_scores[:, 1], pca_scores[:, 2], alpha=0.6, c='blue', edgecolor='k', s=50)
ax3.set_xlabel("PC2")
ax3.set_ylabel("PC3")
ax3.set_xlim([-6000,10000])
ax3.set_ylim([-600,400])
ax3.set_title("Score Plot PC2/PC3")
plt.show(block=False)
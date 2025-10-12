# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Loading the Data
# import libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
df_meta = pd.read_csv(r'D:\HoaiNhan\Code\Code_py\breast-cancer\csv\meta.csv')
df_meta.head()
# load dicom info file
df_dicom = pd.read_csv(r'D:\HoaiNhan\Code\Code_py\breast-cancer\csv\dicom_info.csv')
#df_dicom.head()
# check image types in dataset
df_dicom.SeriesDescription.unique()
# check image path in dataset
# cropped images
cropped_images = df_dicom[df_dicom.SeriesDescription=='cropped images'].image_path
#cropped_images.head(5)
#full mammogram images
full_mammo = df_dicom[df_dicom.SeriesDescription=='full mammogram images'].image_path
#full_mammo.head(5)
# ROI images
roi_img = df_dicom[df_dicom.SeriesDescription=='ROI mask images'].image_path
#roi_img.head(5)
# set correct image path for image types
imdir = 'D:/HoaiNhan/Code/Code_py/breast-cancer/jpeg'
# change directory path of images
cropped_images = cropped_images.replace('CBIS-DDSM/jpeg', imdir, regex=True)
full_mammo = full_mammo.replace('CBIS-DDSM/jpeg', imdir, regex=True)
roi_img = roi_img.replace('CBIS-DDSM/jpeg', imdir, regex=True)
# view new paths
print('Cropped Images paths:\n')
print(cropped_images.iloc[0])
print('Full mammo Images paths:\n')
print(full_mammo.iloc[0])
print('ROI Mask Images paths:\n')
print(roi_img.iloc[0])
# organize image paths
full_mammo_dict = dict()
cropped_images_dict = dict()
roi_img_dict = dict()
for dicom in full_mammo:
    key = dicom.split("/")[-2]
    full_mammo_dict[key] = dicom
for dicom in cropped_images:
    key = dicom.split("/")[-2]
    cropped_images_dict[key] = dicom
for dicom in roi_img:
    key = dicom.split("/")[-2]
    roi_img_dict[key] = dicom

# view keys
next(iter((full_mammo_dict.items())))
# Mass Dataset
# load the mass dataset
mass_train = pd.read_csv(r'D:\HoaiNhan\Code\Code_py\breast-cancer\csv\mass_case_description_train_set.csv')
mass_test = pd.read_csv(r'D:\HoaiNhan\Code\Code_py\breast-cancer\csv\mass_case_description_test_set.csv')

#mass_train.head()
# fix image paths
def fix_image_path(data):
    """correct dicom paths to correct image paths"""
    for index, img in enumerate(data.values):
        img_name = img[11].split("/")[2]
        data.iloc[index,11] = full_mammo_dict[img_name]

        img_name = img[12].split("/")[2]
        data.iloc[index,12] = cropped_images_dict[img_name]

        img_name = img[13].split("/")[2]
        data.iloc[index,13] = roi_img_dict[img_name]
# apply to datasets
fix_image_path(mass_train)
fix_image_path(mass_test)
# check unique values in pathology column
mass_train.pathology.unique()
mass_train.info()
# rename columns
mass_train = mass_train.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

mass_train.head(5)
# check for null values
mass_train.isnull().sum()
# fill in missing values using the backwards fill method
mass_train['mass_shape'] = mass_train['mass_shape'].fillna(method='bfill')
mass_train['mass_margins'] = mass_train['mass_margins'].fillna(method='bfill')

#check null values
#mass_train.isnull().sum()
# quantitative summary of features
mass_train.describe()
# view mass_test
mass_test.head()
# check datasets shape
print(f'Shape of mass_train: {mass_train.shape}')
print(f'Shape of mass_test: {mass_test.shape}')
mass_test.isnull().sum()
# check for column names in mass_test
#print(mass_test.columns)
#print('\n')
# rename columns
mass_test = mass_test.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

# view renamed columns
#mass_test.columns
# fill in missing values using the backwards fill method
mass_test['mass_margins'] = mass_test['mass_margins'].fillna(method='bfill')

#check null values
#mass_test.isnull().sum()
# Visualizations
# pathology distributions
value = mass_train['pathology'].value_counts()
plt.figure(figsize=(8,6))

plt.pie(value, labels=value.index, autopct='%1.1f%%')
plt.title('Breast Cancer Mass Types', fontsize=14)
#plt.savefig('/kaggle/working/pathology_distributions_red.png')
plt.show()
# examine breast assessment types
plt.figure(figsize=(8,6))
sns.countplot(mass_train, y='assessment', hue='pathology', palette='viridis')
plt.title('Breast Cancer Assessment\n\n 0: Undetermined || 1: Well Differentiated\n2: Moderately differentiated || 3: Poorly DIfferentiated\n4-5: Undifferentiated', 
          fontsize=12)
plt.ylabel('Assessment Grade')
plt.xlabel('Count')
#plt.savefig('/kaggle/working/breast_assessment_red.png')
plt.show()
# examine cancer subtlety
plt.figure(figsize=(8,6))
sns.countplot(mass_train, x='subtlety', palette='viridis')
plt.title('Breast Cancer Mass Subtlety', fontsize=12)
plt.xlabel('Subtlety Grade')
plt.ylabel('Count')
#plt.savefig('/kaggle/working/cancer_subtlety_red.png')
plt.show()
# view breast mass shape distribution against pathology
plt.figure(figsize=(8,6))

sns.countplot(mass_train, x='mass_shape', hue='pathology')
plt.title('Mass Shape Distribution by Pathology', fontsize=14)
plt.xlabel('Mass Shape')
plt.xticks(rotation=30, ha='right')
plt.ylabel('Pathology Count')
plt.legend()
#plt.savefig('/kaggle/working/mass_pathology_red.png')
plt.show()
# breast density against pathology
plt.figure(figsize=(8,6))

sns.countplot(mass_train, x='breast_density', hue='pathology')
plt.title('Breast Density vs Pathology\n\n1: fatty || 2: Scattered Fibroglandular Density\n3: Heterogenously Dense || 4: Extremely Dense',
          fontsize=14)
plt.xlabel('Density Grades')
plt.ylabel('Count')
plt.legend()
#plt.savefig('/kaggle/working/density_pathology_red.png')
plt.show()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_images(column, number):
    """displays images in dataset"""
    # Lọc trước những ảnh MALIGNANT
    malignant_rows = mass_train[mass_train['pathology'] == "MALIGNANT"].head(number)
    
    # Tạo figure
    rows = 1
    cols = len(malignant_rows)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    
    # Nếu chỉ có 1 ảnh thì axes không phải mảng
    if cols == 1:
        axes = [axes]
    
    # Loop qua từng ảnh MALIGNANT
    for ax, (_, row) in zip(axes, malignant_rows.iterrows()):
        image_path = row[column]
        image = mpimg.imread(image_path)
        ax.imshow(image, cmap='gray')
        ax.set_title(f"{row['pathology']}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

print('Full Mammograms:\n')
display_images('image_file_path', 5)

print('Cropped Mammograms:\n')
display_images('cropped_image_file_path', 5)

print('ROI Masks:\n')
display_images('ROI_mask_file_path', 5)
# Preprocessing of Images
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def image_processor(image_path, target_size):
    """Preprocess images for CNN model"""
    absolute_image_path = os.path.abspath(image_path)
    image = cv2.imread(absolute_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size[1], target_size[0]))
    image_array = image / 255.0
    return image_array

# Merge datasets
full_mass = pd.concat([mass_train, mass_test], axis=0)

# Define the target size
target_size = (224, 224, 3)

# Apply preprocessor to train data
full_mass['processed_images'] = full_mass['image_file_path'].apply(lambda x: image_processor(x, target_size))

# Create a binary mapper
class_mapper = {'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0} 

# Convert the processed_images column to an array
X_resized = np.array(full_mass['processed_images'].tolist())

# In mảng ảnh đầu tiên
print(X_resized[0])
print("Shape:", X_resized[0].shape)

# Vẽ ảnh đầu tiên
plt.imshow(X_resized[0])
plt.title(full_mass['pathology'][0])
plt.axis('off')
plt.show()

# Apply class mapper to pathology column
full_mass['labels'] = full_mass['pathology'].replace(class_mapper)

# Check the number of classes
num_classes = len(full_mass['labels'].unique())

# Split data into train, test, and validation sets (70, 20, 10)
X_train, X_temp, y_train, y_temp = train_test_split(X_resized, full_mass['labels'].values, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Convert integer labels to one-hot encoded labels
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)
# y_val = to_categorical(y_val, num_classes)
import numpy as np

def gwo_feature_selection_cnn(X, y, cnn_model_builder, num_wolves=8, max_iter=10, feature_ratio=0.5):
    """
    GWO cho chọn feature tối ưu cho CNN.
    X: feature vector (n_samples, n_features)
    y: label
    cnn_model_builder: hàm trả về model CNN (input_shape, num_classes) -> model
    feature_ratio: tỉ lệ feature muốn giữ lại (ví dụ 0.5 là giữ 50%)
    """
    n_features = X.shape[1]
    n_selected = int(n_features * feature_ratio)
    wolves = np.zeros((num_wolves, n_features), dtype=int)
    for i in range(num_wolves):
        wolves[i, np.random.choice(n_features, n_selected, replace=False)] = 1

    alpha, beta, delta = np.zeros(n_features), np.zeros(n_features), np.zeros(n_features)
    alpha_score, beta_score, delta_score = -np.inf, -np.inf, -np.inf

    def fitness(solution):
        idx = np.where(solution == 1)[0]
        if len(idx) == 0:
            return 0
        X_sel = X[:, idx]
        # Build & train model nhỏ (ví dụ 3 epochs, early stop)
        model = cnn_model_builder((X_sel.shape[1],), len(np.unique(y)))
        model.fit(X_sel, y, epochs=3, batch_size=32, verbose=0)
        acc = model.evaluate(X_sel, y, verbose=0)[1]
        return acc

    for it in range(max_iter):
        for i in range(num_wolves):
            score = fitness(wolves[i])
            if score > alpha_score:
                alpha_score, alpha = score, wolves[i].copy()
            elif score > beta_score:
                beta_score, beta = score, wolves[i].copy()
            elif score > delta_score:
                delta_score, delta = score, wolves[i].copy()
        a = 2 - it * (2 / max_iter)
        for i in range(num_wolves):
            for j in range(n_features):
                r1, r2 = np.random.rand(2)
                A = 2 * a * r1 - a
                C = 2 * r2
                D_alpha = abs(C * alpha[j] - wolves[i, j])
                X1 = alpha[j] - A * D_alpha
                wolves[i, j] = 1 if X1 >= 0.5 else 0
        print(f"GWO iter {it+1}/{max_iter}, best acc: {alpha_score:.4f}")
    return np.where(alpha == 1)[0], alpha_score

# --- Ví dụ sử dụng ---
# Giả sử bạn đã có feature vector từ CNN (ví dụ: output của Flatten layer)
# X_feat = model.predict(X_train)  # (n_samples, n_features)
# idx_selected, best_acc = gwo_feature_selection_cnn(X_feat, y_train, cnn_model_builder)
# X_feat_selected = X_feat[:, idx_selected]
# CNN Architecture
# Import necessary TensorFlow libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

# Augment data
train_datagen = ImageDataGenerator(rotation_range=40, 
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True, 
                                   fill_mode='nearest'
                                  )

# apply augmentation to training data
train_data_augmented = train_datagen.flow(X_train, y_train, batch_size=16)

# instantiate CNN model
model = Sequential()

# add layers
model.add(Conv2D(16, (3, 3), activation='relu', 
                 input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten()) # flatten feature maps
model.add(Dense(64, activation='relu')) # add fully connected layers
model.add(Dense(units= 1, activation='sigmoid')) # output layer

# compile model
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# train model
history = model.fit(train_data_augmented, 
                    epochs=20, 
                    validation_data=(X_val, y_val), 
                   )
# model summary
model.summary()
# Evaluation
model.evaluate(X_test, y_test)
model.save('breast_cancer_cnn_model.h5')
# ...existing code...
from keras.models import Model

# Lấy input từ layer đầu tiên thay vì model.input
input_tensor = model.layers[0].input
flatten_layer = model.get_layer(index=-4)
feature_extractor = Model(inputs=input_tensor, outputs=flatten_layer.output)

X_train_feat = feature_extractor.predict(X_train)
X_val_feat = feature_extractor.predict(X_val)
X_test_feat = feature_extractor.predict(X_test)
# ...existing code...
flatten_layer = model.get_layer('flatten_2')
feature_extractor = Model(inputs=model.layers[0].input, outputs=flatten_layer.output)

X_train_feat = feature_extractor.predict(X_train)
X_val_feat = feature_extractor.predict(X_val)
X_test_feat = feature_extractor.predict(X_test)

# Đảm bảo là 2D
X_train_feat = X_train_feat.reshape(X_train_feat.shape[0], -1)
X_val_feat = X_val_feat.reshape(X_val_feat.shape[0], -1)
X_test_feat = X_test_feat.reshape(X_test_feat.shape[0], -1)
def cnn_model_builder(input_shape, num_classes=1):
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Dropout

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Dense(64, activation='relu'))  # fully connected layer
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # output layer cho nhị phân

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


# Chạy GWO
idx_selected, best_acc = gwo_feature_selection_cnn(X_train_feat, y_train, cnn_model_builder, num_wolves=8, max_iter=10, feature_ratio=0.5)
print("Feature index được chọn:", idx_selected)
# Lấy feature đã chọn
X_train_gwo = X_train_feat[:, idx_selected]
X_val_gwo = X_val_feat[:, idx_selected]
X_test_gwo = X_test_feat[:, idx_selected]

# Train lại model nhỏ (hoặc dùng classifier khác như SVM, RF)
model_gwo = cnn_model_builder((X_train_gwo.shape[1],), 1)
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# Khi fit:
history_gwo = model_gwo.fit(X_train_gwo, y_train, epochs=50, validation_data=(X_val_gwo, y_val), callbacks=[early_stop])
test_loss, test_acc = model_gwo.evaluate(X_test_gwo, y_test)
print("Test accuracy (GWO feature):", test_acc)
# Lưu model sau khi train
model_gwo.save('model_gwo_selected_feature.h5')
# ...existing code...

import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(224, 224, 3)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size[1], target_size[0]))
    image = image / 255.0
    return image

# Đường dẫn ảnh cần dự đoán
img_path = r'D:\HoaiNhan\Code\Code_py\breast-cancer\test_image.png'  # <-- thay bằng đường dẫn ảnh thực tế

# Tiền xử lý ảnh
img = preprocess_image(img_path)
plt.imshow(img)
plt.title("Ảnh đầu vào")
plt.axis('off')
plt.show()

# Chuyển ảnh thành vector đặc trưng (feature vector)
img_feat = feature_extractor.predict(img.reshape(1, *img.shape))
img_feat_gwo = img_feat[:, idx_selected]

# Dự đoán

# ...existing code...
# loaded_model = load_model('breast_cancer_cnn_model.h5')
# pred_prob = loaded_model.predict(img.reshape(1, *img.shape))
# ...existing code...
loaded_model = load_model('model_gwo_selected_feature.h5')
pred_prob = loaded_model.predict(img_feat_gwo)
pred_label = int(pred_prob[0, 0] > 0.5)
print("Xác suất dự đoán:", pred_prob[0, 0])
print("Dự đoán label:", pred_label, "|", "MALIGNANT" if pred_label == 1 else "BENIGN")
# ...existing code...
# Classification Report
from sklearn.metrics import classification_report, confusion_matrix

# create labels for confusion matrix
cm_labels = ['MALIGNANT', 'BENIGN']

# obtain predictions
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# convert predicted probabilities to class predictions
y_pred_classes_test = np.argmax(y_pred_test, axis=1)
y_pred_classes_train = np.argmax(y_pred_train, axis=1)

# Assuming y_test and y_val are in binary format (0 or 1)
y_true_classes_test = np.argmax(y_test, axis=1)
y_true_classes_train = np.argmax(y_train, axis=1)

# generate classification reports for test and val sets
test_report = classification_report(y_true_classes_test, y_pred_classes_test, target_names=cm_labels)
train_report = classification_report(y_true_classes_train, y_pred_classes_train, target_names=cm_labels)

# generate confusion matrices for test and validation sets
test_cm = confusion_matrix(y_true_classes_test, y_pred_classes_test)
train_cm = confusion_matrix(y_true_classes_train, y_pred_classes_train)

# create function to print confusion matrix
def plot_confusion_matrix(cm, labels, title):
    """plots a normalized confusion matrix as a heatmap."""
    # Calculate row sums
    row_sums = cm.sum(axis=1, keepdims=True)
    # Normalize confusion matrix
    normalized_cm = cm / row_sums

    plt.figure(figsize=(8, 6))
    sns.heatmap(normalized_cm, annot=True, fmt='.2%', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# print Train and Test reports and matrices
print(f"Train Set Classification report:\n {train_report}\n")
plot_confusion_matrix(train_cm, cm_labels, 'Train Set Confusion Matrix')
print(f"Test Set Classification report:\n {test_report}\n")
plot_confusion_matrix(test_cm, cm_labels, 'Test Set Confusion Matrix')
# ROC_AUC Curves
from sklearn.metrics import roc_curve, auc

# Use the trained model to predict probabilities for the test set
y_pred_prob = model.predict(X_test)

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Print the AUC score
print(f'AUC: {roc_auc:.2f}')
# Visualizing Loss vs Epoch/Accuracy vs Epoch 
history_dict = history.history
# plot training loss vs validation loss
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['accuracy']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'b', label='Training Loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation Loss')
plt.title('Training and Validation Loss', fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#history_df = pd.DataFrame(history.history)
#history_df[['loss', 'val_loss']].plot()

#history_df = pd.DataFrame(history.history)
#history_df[['accuracy', 'val_accuracy']].plot()
# plot training vs validation accuracy
val_acc_values = history_dict['val_accuracy']
acc = history_dict['accuracy']

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc_values, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy', fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Transfer Learning-Hyperparameter Tuning
# model summary
vgg_model.summary()
# Classification Report: Transfer Learning
# classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

# create labels for confusion matrix
cm_labels = ['MALIGNANT', 'BENIGN']

#obtain predictions
y_pred_train_vgg = vgg_model.predict(X_train)
y_pred_test_vgg = vgg_model.predict(X_test)

# convert predicted probabilities to class predictions
y_pred_classes_test_vgg = np.argmax(y_pred_test_vgg, axis=1)
y_pred_classes_train_vgg = np.argmax(y_pred_train_vgg, axis=1)

# get true classes
y_true_classes_train_vgg = np.argmax(y_train, axis=1)
y_true_classes_test_vgg = np.argmax(y_test, axis=1)

# create function to print confusion matrix
def plot_confusion_matrix(cm, labels, title):
    """plots a normalized confusion matrix as a heatmap."""
    # Calculate row sums
    row_sums = cm.sum(axis=1, keepdims=True)
    # Normalize confusion matrix
    normalized_cm = cm / row_sums

    plt.figure(figsize=(8, 6))
    sns.heatmap(normalized_cm, annot=True, fmt='.2%', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# generate classification report
test_report_vgg = classification_report(y_true_classes_test_vgg, y_pred_classes_test_vgg, target_names=cm_labels)
train_report_vgg = classification_report(y_true_classes_train_vgg, y_pred_classes_train_vgg, target_names=cm_labels)

# generate confusion matrix
test_cm_vgg = confusion_matrix(y_true_classes_test_vgg, y_pred_classes_test_vgg)
train_cm_vgg = confusion_matrix(y_true_classes_train_vgg, y_pred_classes_train_vgg)
print(f'Train Set Classifcation report:\n {train_report_vgg}\n')
plot_confusion_matrix(train_cm_vgg, cm_labels, 'Train Set Confusion Matrix: VGG19')
print(f"Test Set Classification report:\n {test_report_vgg}\n")
plot_confusion_matrix(test_cm_vgg, cm_labels, 'Test Set Confusion Matrix: VGG19')
# ROC-AUC Curves: Transfer Learning
# ROC-AUC Curves
from sklearn.metrics import roc_curve, auc

# Use the trained model to predict probabilities for the test set
y_pred_prob = vgg_model.predict(X_test)

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic-Area Under Curve (ROC-AUC)')
plt.legend(loc='lower right')
plt.show()

# Print the AUC score
print(f'AUC: {roc_auc:.2f}')
# Epochs-Loss-Accuracy Visualization: Transfer Learning
pre_train_dict = history_3.history
# plot training loss vs validation loss
loss_values = pre_train_dict['loss']
val_loss_values = pre_train_dict['val_loss']
acc = pre_train_dict['accuracy']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'b', label='Training Loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation Loss')
plt.title('Training and Validation Loss', fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# plot training vs validation accuracy
val_acc_values = pre_train_dict['val_accuracy']

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc_values, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy', fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Human Readable Predictions
predictions = vgg_model.predict(X_test)
import random

# reverse class mapping
reverse_mapper = {v:k for k, v in class_mapper.items()}

# map predictions to class_names
predicted_class_indices = np.argmax(predictions, axis=1)
predicted_class_names = [reverse_mapper[i] for i in predicted_class_indices]

ground_truth_class_indices = np.argmax(y_test, axis=1)
ground_truth_class_names = [reverse_mapper[i] for i in ground_truth_class_indices]
# display predicted class_names
num_image_visualize = min(5, len(X_test))

# create random indices to select images
random_indices = random.sample(range(len(X_test)), num_image_visualize)

# create subplots for images
fig, ax = plt.subplots(1, num_image_visualize, figsize=(15, 5))

for i, idx in enumerate(random_indices):
    ax[i].imshow(X_test[idx])
    ax[i].set_title(f'Predicted: {predicted_class_names[idx]}', fontsize=10, color='red')
    ax[i].text(0.5, -0.1, f'Truth: {ground_truth_class_names[idx]}', fontsize=10, ha='center', va='center', 
              transform=ax[i].transAxes, color='blue')
    ax[i].axis('off')

plt.tight_layout()
plt.show()
# Save Model
vgg_model.save('transfer_learning-1_model.h5')
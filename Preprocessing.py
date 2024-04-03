from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
width_shift_range=0.2,
height_shift_range=0.2,
fill_mode='nearest';)
valid_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 128
base_dir = '../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New PlantDiseases Dataset(Augmented)';
training_set = train_datagen.flow_from_directory(base_dir+'/train',
target_size=(224, 224),
batch_size=batch_size,
class_mode='categorical',
shuffle=False)
valid_set = valid_datagen.flow_from_directory(base_dir+'/valid',
target_size=(224, 224),
batch_size=batch_size,
class_mode='categorical',
shuffle=False)

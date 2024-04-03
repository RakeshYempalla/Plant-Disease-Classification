import time
from keras.callbacks import ModelCheckpoint
start = time.time()
weightpath = "best_weights_9.hdf5"
checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1,
save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [checkpoint]
#fitting images to CNN
history = classifier.fit_generator(training_set,
steps_per_epoch=150, #train_num//batch_size,
validation_data=valid_set,
epochs=18, #25,
validation_steps=100, #valid_num//batch_size,
callbacks=callbacks_list)
#saving model
filepath='AlexNetModel.hdf5';
classifier.save(filepath)
end = time.time()
print('\nTime taken for training the dataset:',(end - start)/60, 'minutes')

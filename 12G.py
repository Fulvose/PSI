model_transfer.compile(loss='binary_crossentropy',optimizer="sgd",metrics=['accuracy'])

train_data_dir = './Dane/data/PetImagesTrain'
validation_data_dir = './Dane/data/validation'
nb_validation_samples = 200
nb_train_samples = 50
epochs = 50
batch_size = 10

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.2, 
                                   zoom_range=0.2,
                                   rotation_range=45, 
                                   horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir, 
                                                    target_size=(h, w), 
                                                    batch_size=batch_size, 
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(h, w), 
                                                        batch_size=batch_size,
                                                        class_mode='binary')

history_tr_ag2 = History()
early_stopping = EarlyStopping(patience=3,monitor="val_loss")
model_transfer.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs, 
                    validation_data=validation_generator, validation_steps=10, callbacks=[early_stopping, history_tr_ag2])
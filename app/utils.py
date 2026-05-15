import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to get correct image path
def get_image_path(image_id, part1_path, part2_path):
    filename = image_id + ".jpg"
    
    path1 = os.path.join(part1_path, filename)
    
    if os.path.exists(path1):
        return path1
    else:
        return os.path.join(part2_path, filename)


# Function to create generators
def get_generators(train_df, val_df, test_df):

    # Data augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Only rescaling for validation and test
    val_test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    # Train generator
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw'
    )

    # Validation generator
    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw'
    )

    # Test generator
    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='image_path',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        shuffle=False
    )

    return train_generator, val_generator, test_generator
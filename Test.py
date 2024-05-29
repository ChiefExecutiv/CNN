import keras
from keras.utils import image_dataset_from_directory



train_dir = "c:\\Users\\DELL\\Desktop\\Brain_Dataset\\train"
val_dir = "C:\\Users\\DELL\\Desktop\\Brain_Dataset\\validate"
test_dir = "c:\\Users\\DELL\\Desktop\\Brain_Dataset\\test"

train_dataset = image_dataset_from_directory(
    train_dir,
    image_size=(180, 180),
    batch_size=8
)

validation_dataset = image_dataset_from_directory(
    val_dir,
    image_size=(180, 180),
    batch_size=8
)

test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=(180, 180),
    batch_size=8
)

for data_batch, labels_batch in train_dataset:
    print(f"data batch shape:, {data_batch.shape}")
    print(f"labels batch shape: {labels_batch.shape}")
    break



test_model = keras.models.load_model("Brain_model_v2.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc}")
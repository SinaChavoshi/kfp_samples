import tensorflow as tf
mnist = tf.keras.datasets.mnist
from tensorflow.python.lib.io import file_io
import argparse

def process_args():
  """Define arguments and assign default values to the ones that are not set.
  Returns:
    args: The parsed namespace with defaults assigned to the flags.
  """

  parser = argparse.ArgumentParser(
      description='Runs MNIST Kubeflow Pipeline Sample E2E.')
  parser.add_argument(
      '--mode',
      default='all',
      help='execution mode, choose between eval, train, or all.'
      'Default is all')
  parser.add_argument(
      '--output_path',
      default=None,
      help='output path for saving the output model file from'
      'training step. same path is used to load the model from'
      'for evaluation step' )
  parser.add_argument(
      '--epochs',
      default=5,
      help='number of epochs to run the training, default is 5.')
  args, _ = parser.parse_known_args()
  return args

def main():
    args = process_args()

    #load training / eval data
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    #setup a temporary location for model file 
    temp_model_location = './temp_model.h5'
        
    #create a new entry point to choose based on execution_mode 
    if args.mode == 'train' or args.mode == 'all':
        
        print('Execution step - model training')
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=int(args.epochs))

        model.save(filepath=temp_model_location)

        # use tf.io to read/write to gs or s3
        temp_model_file = open(temp_model_location, 'rb')
        cloud_model_file = file_io.FileIO(args.output_path, mode='wb')
        cloud_model_file.write(temp_model_file.read())
        temp_model_file.close()
        cloud_model_file.close()        
        
        return 

    # for execution mode = evaluate load a pre trained model
    if args.mode == 'eval' or args.mode == 'all':
        print('Execution step - model evaluation')

        # use tf.io to read/write to gs or s3
        cloud_model_file = file_io.FileIO(args.output_path, mode='rb')
        
        temp_model_file = open(temp_model_location, 'wb')
        temp_model_file.write(cloud_model_file.read())
        temp_model_file.close()
        cloud_model_file.close()
        
        model = tf.keras.models.load_model(filepath=temp_model_location)
        model.evaluate(x_test, y_test)
    
    return 
  
if __name__== "__main__":
  main()

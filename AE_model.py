from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

def outlier_model(img_width, img_height):
  # input layer
  input_layer = Input(shape=(img_width, img_height, 3))

  # encoding architecture
  conv_l1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
  max_l1 = MaxPooling2D( (2, 2), padding='same')(conv_l1)
  conv_l2 = Conv2D(32, (3, 3), activation='relu', padding='same')(max_l1)
  max_l2 = MaxPooling2D( (2, 2), padding='same')(conv_l2)
  conv_l3 = Conv2D(16, (3, 3), activation='relu', padding='same')(max_l2)
  #bottle_neck
  bottle_neck = MaxPooling2D( (2, 2), padding='same')(conv_l3)

  # decoding architecture
  conv_l1 = Conv2D(16, (3, 3), activation='relu', padding='same')(bottle_neck)
  upsamp_l1 = UpSampling2D((2, 2))(conv_l1)
  conv_l2 = Conv2D(32, (3, 3), activation='relu', padding='same')(upsamp_l1)
  upsamp_l2 = UpSampling2D((2, 2))(conv_l2)
  conv_l3 = Conv2D(64, (3, 3), activation='relu')(upsamp_l2)
  upsamp_l3 = UpSampling2D((2, 2))(conv_l3)
  output_layer  = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(upsamp_l3)

  # compile the model
  model = Model(input_layer, output_layer)
  model.compile(optimizer='adam', loss='mse')
  return model
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import sys

#import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()

import util
import biaffine_ner_model
import jsonlines

if __name__ == "__main__":
  config = util.initialize_from_env()

  # config['eval_path'] = config['test_path']
  config['eval_path'] = sys.argv[2]
  output_file = sys.argv[3]

  print(f'Inference on {sys.argv[2]}')
  model = biaffine_ner_model.BiaffineNERModel(config)

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  session_config.allow_soft_placement = True
  with tf.Session(config=session_config) as session:
    model.restore(session)
    predictions = model.predict(session,True)

    with jsonlines.open(output_file, 'w') as writer:
      writer.write_all(predictions)
    # json.dump(predictions, open(output_file, 'w'), indent=4)

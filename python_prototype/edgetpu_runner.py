import glob
import time
import numpy as np
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from argparse import ArgumentParser

NUM_CLIPS_PER_FILE = 500 # Only valid for 256Hz 

def main():
    # Parser
    parser = ArgumentParser(description='Run TensorFlow Lite model on Coral USB Accelerator Edge TPU.')
    parser.add_argument('--model_fp', help='Filepath of .tflite model to run.', type=str, required=True)
    parser.add_argument('--data_dir', help='Directory path of .npy data to use.', type=str, required=True)

    # Parse arguments
    args = parser.parse_args()

    # Prepare model interpreter
    interpreter = tf.lite.Interpreter(*args.model_fp.split('@'))
    interpreter.allocate_tensors()

    # Input data
    input_data_filepaths = glob.glob(args.data_dir + "/*.npy")
    print(input_data_filepaths)

    # Run inference
    inference_times = np.empty((1))
    print('---- INFERENCE ----')
    file_cnt = 0
    results_string = ""

    for filepath in input_data_filepaths:
        input_data = np.load(filepath) # Dimensions (# of clips, 1, clip_length). 1 indicates a batch_size of 1.
    
        for i in range(input_data.shape[0]):
            common.input_tensor(interpreter)[:] = input_data[i]
            start = time.perf_counter()
            interpreter.invoke()
            inference_time = time.perf_counter() - start
            top_output_class = classify.get_classes(interpreter, top_k=1)
            print(f"Clip {NUM_CLIPS_PER_FILE*file_cnt + i} output: {top_output_class[0].id} (inference time: {(inference_time * 1000):.3f}ms)")
            results_string += str(top_output_class[0].id) + '\n'
            inference_times = np.append(inference_times, 1000*inference_time)

        file_cnt += 1

    # Write results to file
    with open(args.data_dir + "/out_tpu.txt", 'w') as text_file:
        text_file.write(f"Model: {args.model_fp}\n")
        text_file.write(f"Mean time: {np.mean(inference_times[2:]):.3f}ms, std. dev.: {np.std(inference_times[2:]):.3f}ms\n")
        text_file.write(results_string)

    # Report stats
    print(f"Done. Mean time: {np.mean(inference_times[2:]):.3f}ms, std. dev.: {np.std(inference_times[2:]):.3f}ms\n")

if __name__ == "__main__":
    main()

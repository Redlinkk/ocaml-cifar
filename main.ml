open Base
open Float.O_dot
open Tensorflow_core
open Tensorflow
module In_channel = Stdio.In_channel
module O = Ops

let image_dim = 3 * 32 * 32
let label_count = 10

let one_hot labels =
  let nsamples = Bigarray.Array1.dim labels in
  let one_hot = Tensor.create2 Float32 nsamples label_count in
  for idx = 0 to nsamples - 1 do
    for lbl = 0 to 9 do
      Tensor.set one_hot [| idx; lbl |] 0.
    done;
    let lbl = Bigarray.Array1.get labels idx |> Int32.to_int_exn in
    Tensor.set one_hot [| idx; lbl |] 1.
  done;
  one_hot

let read_data filename =
  let in_channel = In_channel.create filename in
  let rows = 32 in
  let columns = 32 in
  let channels = 3 in
  let samples = 10000 in
  let images = Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout samples (rows * columns * channels) in
  let labels = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout samples in
  for sample = 0 to samples - 1 do
    let v = Option.value_exn (In_channel.input_byte in_channel) |> Int32.of_int_exn in
    Bigarray.Array1.set labels sample v;
    for idx = 0 to rows * columns * channels - 1 do
      let v = Option.value_exn (In_channel.input_byte in_channel) in
      Bigarray.Array2.set images sample idx Float.(of_int v / 255.);
    done;
  done;
  In_channel.close in_channel;
  images, labels

type float32_tensor = (float, Bigarray.float32_elt) Tensor.t

type t =
  { image_batch_1 : float32_tensor
  ; label_batch_1 : float32_tensor
  ; image_batch_2 : float32_tensor
  ; label_batch_2 : float32_tensor
  ; image_batch_3 : float32_tensor
  ; label_batch_3 : float32_tensor
  ; image_batch_4 : float32_tensor
  ; label_batch_4 : float32_tensor
  ; image_batch_5 : float32_tensor
  ; label_batch_5 : float32_tensor
  ; test_images : float32_tensor
  ; test_labels : float32_tensor
  }

let read_files 
      ?(data_batch_1 = "data/data_batch_1.bin")
      ?(data_batch_2 = "data/data_batch_2.bin")
      ?(data_batch_3 = "data/data_batch_3.bin")
      ?(data_batch_4 = "data/data_batch_4.bin")
      ?(data_batch_5 = "data/data_batch_5.bin")
      ?(test_data = "data/test_batch.bin")
      ()
  =
  let image_batch_1, label_batch_1 = read_data data_batch_1 in
  let image_batch_2, label_batch_2 = read_data data_batch_2 in
  let image_batch_3, label_batch_3 = read_data data_batch_3 in
  let image_batch_4, label_batch_4 = read_data data_batch_4 in
  let image_batch_5, label_batch_5 = read_data data_batch_5 in
  let test_images, test_labels = read_data test_data in
  let image_batch_1 =
    Bigarray.genarray_of_array2 image_batch_1 
    |> Tensor.of_bigarray ~scalar:false 
  in
  let image_batch_2 =
    Bigarray.genarray_of_array2 image_batch_2 
    |> Tensor.of_bigarray ~scalar:false 
  in
  let image_batch_3 =
    Bigarray.genarray_of_array2 image_batch_3 
    |> Tensor.of_bigarray ~scalar:false 
  in
  let image_batch_4 =
    Bigarray.genarray_of_array2 image_batch_4 
    |> Tensor.of_bigarray ~scalar:false 
  in
  let image_batch_5 =
    Bigarray.genarray_of_array2 image_batch_5 
    |> Tensor.of_bigarray ~scalar:false 
  in
  let test_images =
    Bigarray.genarray_of_array2 test_images 
    |> Tensor.of_bigarray ~scalar:false 
  in
  { image_batch_1
  ; label_batch_1 = one_hot label_batch_1
  ; image_batch_2
  ; label_batch_2 = one_hot label_batch_2
  ; image_batch_3
  ; label_batch_3 = one_hot label_batch_3
  ; image_batch_4
  ; label_batch_4 = one_hot label_batch_4
  ; image_batch_5
  ; label_batch_5 = one_hot label_batch_5
  ; test_images
  ; test_labels = one_hot test_labels
  }

let train_batch { image_batch_1; label_batch_1; _ } ~batch_size ~batch_idx =
  let train_size = (Tensor.dims image_batch_1).(0) in
  let start_batch = Int.(%) (batch_size * batch_idx) (train_size - batch_size) in
  let batch_images = Tensor.sub_left image_batch_1 start_batch batch_size in
  let batch_labels = Tensor.sub_left label_batch_1 start_batch batch_size in
  batch_images, batch_labels

let accuracy ys ys' =
  let ys = Bigarray.array2_of_genarray (Tensor.to_bigarray ys) in
  let ys' = Bigarray.array2_of_genarray (Tensor.to_bigarray ys') in
  let nsamples = Bigarray.Array2.dim1 ys in
  let res = ref 0. in
  let find_best_idx ys n =
    let best_idx = ref 0 in
    for l = 1 to label_count - 1 do
      let v = Bigarray.Array2.get ys n !best_idx in
      let v' = Bigarray.Array2.get ys n l in
      if Float.(>) v' v then best_idx := l
    done;
    !best_idx
  in
  for n = 0 to nsamples - 1 do
    let idx = find_best_idx ys n in
    let idx' = find_best_idx ys' n in
    res := Float.(+) !res (if idx = idx' then 1. else 0.)
  done;
  Float.(!res / of_int nsamples)

let batch_accuracy ?samples t train_or_test ~batch_size ~predict =
  let images, labels =
    match train_or_test with
    | `train -> t.image_batch_1, t.label_batch_1
    | `test -> t.test_images, t.test_labels
  in
  let dataset_samples = (Tensor.dims labels).(0) in
  let samples =
    Option.value_map samples ~default:dataset_samples ~f:(Int.min dataset_samples)
  in
  let rec loop start_index sum_accuracy =
    if samples <= start_index
    then sum_accuracy /. Float.of_int samples
    else
      let batch_size = Int.min batch_size (samples - start_index) in
      let images = Tensor.sub_left images start_index batch_size in
      let predicted_labels = predict images in
      let labels = Tensor.sub_left labels start_index batch_size in
      let batch_accuracy = accuracy predicted_labels labels in
      loop
        (start_index + batch_size)
        (sum_accuracy +. batch_accuracy *. Float.of_int batch_size)
  in
  loop 0 0.

let scalar_tensor f =
  let array = Tensor.create1 Bigarray.float32 1 in
  Tensor.set array [| 0 |] f;
  array

let batch_size = 512
let epochs = 5000

let () =
  let cifar = read_files () in
  let keep_prob = O.placeholder [] ~type_:Float in
  let xs = O.placeholder [-1; image_dim] ~type_:Float in
  let ys = O.placeholder [-1; label_count] ~type_:Float in

  let ys_ =
    O.Placeholder.to_node xs
    |> Layer.reshape ~shape:[ -1; 32; 32; 3 ]
    |> Layer.conv2d ~ksize:(5, 5) ~strides:(1, 1) ~output_dim:32
    |> Layer.max_pool ~ksize:(2, 2) ~strides:(2, 2)
    |> Layer.conv2d ~ksize:(5, 5) ~strides:(1, 1) ~output_dim:64
    |> Layer.max_pool ~ksize:(2, 2) ~strides:(2, 2)
    |> Layer.flatten
    |> Layer.linear ~output_dim:1024 ~activation:Relu
    |> O.dropout ~keep_prob:(O.Placeholder.to_node keep_prob)
    |> Layer.linear ~output_dim:10 ~activation:Softmax
  in

  let cross_entropy = O.cross_entropy ~ys:(O.Placeholder.to_node ys) ~y_hats:ys_ `sum in
  let gd = Optimizers.adam_minimizer ~learning_rate:(O.f 1e-5) cross_entropy in
  let one = scalar_tensor 1. in
  let predict images =
    Session.run (Session.Output.float ys_)
      ~inputs:Session.Input.[ float xs images; float keep_prob one ]
  in
  let print_err n =
    let test_accuracy =
      batch_accuracy cifar `test ~batch_size:1024 ~predict
    in
    let train_accuracy =
      batch_accuracy cifar `train ~batch_size:1024 ~predict ~samples:5000
    in
    Stdio.printf "epoch %d, train: %.2f%% valid: %.2f%%\n%!"
      n (100. *. train_accuracy) (100. *. test_accuracy)
  in
  let half = scalar_tensor 0.5 in
  for batch_idx = 1 to epochs do
    let batch_images, batch_labels =
      train_batch cifar ~batch_size ~batch_idx
    in
    if batch_idx % 100 = 0 then print_err batch_idx;
    Session.run
      ~inputs:Session.Input.[
        float xs batch_images; float ys batch_labels; float keep_prob half ]
      ~targets:gd
      Session.Output.empty;
  done

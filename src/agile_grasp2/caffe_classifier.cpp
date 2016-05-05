#include <agile_grasp2/caffe_classifier.h>


Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& label_file)
{
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}


/* Return the predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, bool use_softmax)
{
  std::vector<float> output = Predict(img, use_softmax);
  std::vector<Prediction> predictions(output.size());
  for (int i = 0; i < output.size(); ++i)
  {
    predictions[i] = std::make_pair(labels_[i], output[i]);
  }

  return predictions;
}


std::vector<float> Classifier::Predict(const cv::Mat& img, bool use_softmax)
{
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  const vector<Blob<float>*>& result = net_->ForwardPrefilled();
//  std::cout << result.size() << "\n";
//  const float* prob_vec = result[0]->cpu_data();
//  std::cout << "prob_vec: " << prob_vec[0] << " " << prob_vec[1] << std::endl;

  /* Copy the output layer to a std::vector */

  boost::shared_ptr<caffe::Blob<float> > output_layer;
  if (use_softmax)
    output_layer = net_->blob_by_name("prob");
  else
    output_layer = net_->blob_by_name("ip2");
//  const boost::shared_ptr<caffe::Blob<float> > output_layer = net_->blob_by_name("ip2");
//  std::cout << output_layer->shape_string() << std::endl;
//  std::cout << net_->blobs()[4]->shape_string() << std::endl;
//  predictionProbs = net.blobs['ip2'].data[0].copy()
//      prediction = net.blobs['ip2'].data[0].argmax(0)
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();

//  std::cout << "output_layer->cpu_data(): " << begin[0] << std::endl;
//  std::cout << "output_layer->cpu_data(): " << begin[1] << std::endl;
//  std::cout << "output_layer->cpu_data(): " << (1.0)/(1.0 + exp(-1.0*begin[0])) << std::endl;
//  std::cout << "output_layer->cpu_data(): " << (1.0)/(1.0 + exp(-1.0*begin[1])) << std::endl;
//
//  const shared_ptr<Blob<float> >& probs = net_->blob_by_name("prob");
//  const float* probs_out = probs->cpu_data();
//  std::cout << "probs: " << probs_out[0] << " " << probs_out[1] << "\n";

//  Blob<float>* outputs = net_->output_blobs()[0];
//  const float* begin2 = output_layer->cpu_data();
//  std::cout << "outputs->cpu_data(): " << begin2[0] << std::endl;
//  std::cout << "outputs->cpu_data(): " << begin2[1] << std::endl;

//  const float* data = output_layer->data();
//  std::cout << "output_layer->data(): " << data[0] << std::endl;
//  std::cout << "output_layer->data(): " << data[1] << std::endl;

  return std::vector<float>(begin, end);
}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}


void Classifier::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels)
{
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized = sample_float;
//  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

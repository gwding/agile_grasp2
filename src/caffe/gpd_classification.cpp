#include <boost/filesystem.hpp>
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& label_file) {
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

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);
//  std::cout << "Predict (img) ...\n";
//  for (int i = 0; i < output.size(); i++)
//  {
//    std::cout << i << ", output: " << output[i] << std::endl;
//  }

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
//    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[i], output[i]));
//    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
//  const Blob<float>* output_layer = net_->output_blobs()[0];
  const boost::shared_ptr<caffe::Blob<float> > output_layer = net_->blob_by_name("ip2");
//  std::cout << output_layer->shape_string() << std::endl;
//  std::cout << net_->blobs()[4]->shape_string() << std::endl;
//  predictionProbs = net.blobs['ip2'].data[0].copy()
//      prediction = net.blobs['ip2'].data[0].argmax(0)
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
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

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
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

int main(int argc, char** argv) {
  if (argc != 5)
  {
    std::cerr << "Usage: " << argv[0] << " deploy.prototxt network.caffemodel labels.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string label_file   = argv[3];
  Classifier classifier(model_file, trained_file, label_file);

  boost::filesystem::path img_dir(argv[4]);

  if (boost::filesystem::exists(img_dir))    // does p actually exist?
  {
    if (is_regular_file(img_dir))        // is p a regular file?
    {
      std::cout << "Classifying single image ...\n";
      cv::Mat img = cv::imread(argv[4], -1);
      CHECK(!img.empty()) << "Unable to decode image " << argv[4];
      std::vector<Prediction> predictions = classifier.Classify(img);

      /* Print the top N predictions. */
      for (size_t i = 0; i < predictions.size(); ++i) {
       Prediction p = predictions[i];
       std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                 << p.first << "\"" << std::endl;
      }
    }
    else if (is_directory(img_dir))      // is p a directory?
    {
//      std::cout << " is a directory containing:\n";
//      std::vector<boost::filesystem::directory_entry> entries;
//      std::copy(boost::filesystem::directory_iterator(img_dir), boost::filesystem::directory_iterator(), std::back_inserter(entries));
//      for (std::vector<boost::filesystem::directory_entry>::const_iterator it = entries.begin(); it != entries.end();  it++)
//      {
//        std::cout << (*it).path().string() << std::endl;
//      }
      std::string img_list_file = img_dir.string() + "test.txt";
      std::cout << "Processing images in directory: " << img_list_file << std::endl;
      std::ifstream infile(img_list_file.c_str());
      std::string str;
      std::vector< std::vector<float> > scores_list;
      while (std::getline(infile, str))
      {
        std::string imgfile = img_dir.string() + "jpgs/" + str;
        cv::Mat img = cv::imread(imgfile, -1);
        CHECK(!img.empty()) << "Unable to decode image " << imgfile;
        std::vector<Prediction> predictions = classifier.Classify(img);
        std::cout << str << ": ";
        std::vector<float> scores;
        for (size_t i = 0; i < predictions.size(); ++i)
        {
          Prediction p = predictions[i];
          std::cout << std::fixed << std::setprecision(4) << p.second << " ";
          scores.push_back(p.second);
        }
//        std::cout << img.type << " ";
        std::cout << std::endl;
        scores_list.push_back(scores);
      }
    }
    else
    {
      std::cout << img_dir << " exists, but is neither a regular file nor a directory\n";
    }
  }

  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV

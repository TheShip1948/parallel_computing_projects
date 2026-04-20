#ifndef LAYER_CONVOLUTIONAL_H_
#define LAYER_CONVOLUTIONAL_H_

#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include "../Config.h"
#include "../Layer.h"
#include "../Utils/Convolution.h"
#include "../Utils/Random.h"
#include "../Utils/IO.h"
#include "../Utils/Enum.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>


namespace MiniDNN
{

    static __global__ void convolve_kernel(int nobs, int in_channels, int out_channels,
                                          int in_h, int in_w, int k_h, int k_w,
                                          int out_h, int out_w,
                                          const Scalar* __restrict__     input, 
                                          const Scalar* __restrict__     weights, 
                                          Scalar* __restrict__     output)
    {
        int tx = blockIdx.x * blockDim.x + threadIdx.x; // maps to out_h * out_w (flat)
        int ty = blockIdx.y * blockDim.y + threadIdx.y; // maps to nobs * out_channels (flat)

        if (tx < out_h * out_w && ty < nobs * out_channels)
        {
            int n = ty / out_channels;
            int out_c = ty % out_channels;
            int row = tx % out_h; // Column-major: index % rows is row
            int col = tx / out_h; // Column-major: index / rows is col

            Scalar sum = 0.0;
            for (int in_c = 0; in_c < in_channels; in_c++)
            {
                const Scalar* cur_in = input + n * (in_channels * in_h * in_w) + in_c * (in_h * in_w);
                const Scalar* cur_w = weights + in_c * (out_channels * k_h * k_w) + out_c * (k_h * k_w);

                for (int i = 0; i < k_h; i++)
                {
                    for (int j = 0; j < k_w; j++)
                    {
                        int in_row = row + i;
                        int in_col = col + j;
                        // Eigen layout: in_col * in_h + in_row
                        sum += cur_in[in_col * in_h + in_row] * cur_w[j * k_h + i];
                    }
                }
            }
            int output_idx = n * (out_channels * out_h * out_w) + out_c * (out_h * out_w) + col * out_h + row;
            output[output_idx] = sum;
        }
    }


///
/// \ingroup Layers
///
/// Convolutional hidden layer
///
/// Currently only supports the "valid" rule of convolution.
///
template <typename Activation>
class Convolutional: public Layer
{
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Matrix::ConstAlignedMapType ConstAlignedMapMat;
        typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
        typedef Vector::AlignedMapType AlignedMapVec;
        typedef std::map<std::string, int> MetaInfo;

        const internal::ConvDims m_dim; // Various dimensions of convolution


        Vector m_filter_data;  // Filter parameters. Total length is
                               // (in_channels x out_channels x filter_rows x filter_cols)
                               // See Utils/Convolution.h for its layout

        Vector m_df_data;      // Derivative of filters, same dimension as m_filter_data

        Vector m_bias;         // Bias term for the output channels, out_channels x 1. (One bias term per channel)
        Vector m_db;           // Derivative of bias, same dimension as m_bias

        Matrix m_z;            // Linear term, z = conv(in, w) + b. Each column is an observation
        Matrix m_a;            // Output of this layer, a = act(z)
        Matrix m_din;          // Derivative of the input of this layer
                               // Note that input of this layer is also the output of previous layer
        
    public:
        class ForwardStrategy {
            public:            
                virtual ~ForwardStrategy() = default;           
                virtual void forward(const Matrix& prev_layer_data, Convolutional<Activation>* layer) = 0;
        };

        class MiniDNNForwardStrategy : public ForwardStrategy {
            public:
                virtual ~MiniDNNForwardStrategy() = default;
                virtual void forward(const Matrix& prev_layer_data, Convolutional<Activation>* layer) {
                    // Each column is an observation
                    const int nobs = prev_layer_data.cols(); // Number of images in the batch 
                    // Linear term, z = conv(in, w) + b
                    layer->m_z.resize(layer->m_out_size, nobs);
                    // Convolution
                    auto start = std::chrono::high_resolution_clock::now();
                    internal::convolve_valid(layer->m_dim, prev_layer_data.data(), true, nobs,
                                            layer->m_filter_data.data(), layer->m_z.data()
                                    );
                    auto end = std::chrono::high_resolution_clock::now();
                    layer->m_last_conv_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    // Add bias terms
                    // Each column of m_z contains m_dim.out_channels channels, and each channel has
                    // m_dim.conv_rows * m_dim.conv_cols elements
                    int channel_start_row = 0;
                    const int channel_nelem = layer->m_dim.conv_rows * layer->m_dim.conv_cols;

                    for (int i = 0; i < layer->m_dim.out_channels; i++, channel_start_row += channel_nelem)
                    {
                        layer->m_z.block(channel_start_row, 0, channel_nelem, nobs).array() += layer->m_bias[i];
                    }

                    // Apply activation function
                    layer->m_a.resize(layer->m_out_size, nobs);
                    Activation::activate(layer->m_z, layer->m_a);
                };
        };

        class CPUForwardStrategy : public ForwardStrategy {
            private: 
                void convolve(const Matrix& prev_layer_data, Convolutional<Activation>* layer) {
                const int nobs = prev_layer_data.cols();
                const int in_channels = layer->m_dim.in_channels;
                const int out_channels = layer->m_dim.out_channels;
                const int in_h = layer->m_dim.channel_rows;
                const int in_w = layer->m_dim.channel_cols;
                const int k_h = layer->m_dim.filter_rows;
                const int k_w = layer->m_dim.filter_cols;
                const int out_h = layer->m_dim.conv_rows;
                const int out_w = layer->m_dim.conv_cols;

                layer->m_z.setZero();

                for (int n = 0; n < nobs; n++) { // Iterate over each image or observation in the batch 
                    for (int out_c = 0; out_c < out_channels; out_c++) { // Iterate over each output channel 
                        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
                            out_mat(layer->m_z.data() + n * layer->m_z.rows() + out_c * out_h * out_w, out_h, out_w);

                        for (int in_c = 0; in_c < in_channels; in_c++) { // Iterate over each input channel 
                            Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> 
                                in_mat(prev_layer_data.data() + n * prev_layer_data.rows() + in_c * in_h * in_w, in_h, in_w);
                            Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
                                filter_mat(layer->m_filter_data.data() + in_c * (out_channels * k_h * k_w) + out_c * k_h * k_w, k_h, k_w);

                            for (int i = 0; i < out_h; i++) { // Iterate over each row of the output channel 
                                for (int j = 0; j < out_w; j++) { // Iterate over each column of the output channel
                                    out_mat(i, j) += (in_mat.block(i, j, k_h, k_w).array() * filter_mat.array()).sum();
                                }
                            }
                        }
                    }
                }
                }

            public: 
                virtual ~CPUForwardStrategy() = default; 
                virtual void forward(const Matrix& prev_layer_data, Convolutional<Activation>* layer) override {
                    std::cout << "CPU Forward Strategy" << std::endl;
                    // TODO: Implement CPU forward strategy
                    // Each column is an observation
                    const int nobs = prev_layer_data.cols(); // Number of images in the batch 
                    // Linear term, z = conv(in, w) + b
                    layer->m_z.resize(layer->m_out_size, nobs);
                    
                    // Convolution
                    auto start = std::chrono::high_resolution_clock::now();
                    convolve(prev_layer_data, layer);
                    auto end = std::chrono::high_resolution_clock::now();
                    layer->m_last_conv_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

                    // Add bias terms
                    int channel_start_row = 0;
                    const int channel_nelem = layer->m_dim.conv_rows * layer->m_dim.conv_cols;

                    for (int i = 0; i < layer->m_dim.out_channels; i++, channel_start_row += channel_nelem)
                    {
                        layer->m_z.block(channel_start_row, 0, channel_nelem, nobs).array() += layer->m_bias[i];
                    }

                    // Apply activation function
                    layer->m_a.resize(layer->m_out_size, nobs);
                    Activation::activate(layer->m_z, layer->m_a);
                }
        };

        // TODO: the forward algorithm is same everywhere, the only difference is the convolve function. I need to 
        //       refactor the code so that only to emphasize convolve function  
        class GPUForwardStrategy : public ForwardStrategy {
            private: 
                void convolve(const Matrix& prev_layer_data, Convolutional<Activation>* layer) {
                    const int nobs = prev_layer_data.cols();
                    const int in_channels = layer->m_dim.in_channels;
                    const int out_channels = layer->m_dim.out_channels;
                    const int in_h = layer->m_dim.channel_rows;
                    const int in_w = layer->m_dim.channel_cols;
                    const int k_h = layer->m_dim.filter_rows;
                    const int k_w = layer->m_dim.filter_cols;
                    const int out_h = layer->m_dim.conv_rows;
                    const int out_w = layer->m_dim.conv_cols;

                    Scalar *d_in, *d_w, *d_out;
                    size_t in_numelem = static_cast<size_t>(nobs) * in_channels * in_h * in_w;
                    size_t w_numelem = static_cast<size_t>(in_channels) * out_channels * k_h * k_w;
                    size_t out_numelem = static_cast<size_t>(nobs) * out_channels * out_h * out_w;

                    cudaMalloc(&d_in, in_numelem * sizeof(Scalar));
                    cudaMalloc(&d_w, w_numelem * sizeof(Scalar));
                    cudaMalloc(&d_out, out_numelem * sizeof(Scalar));

                    cudaMemcpy(d_in, prev_layer_data.data(), in_numelem * sizeof(Scalar), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_w, layer->m_filter_data.data(), w_numelem * sizeof(Scalar), cudaMemcpyHostToDevice);
                    cudaMemset(d_out, 0, out_numelem * sizeof(Scalar));

                    // dim3 blockSize(16, 16);
                    dim3 blockSize(32, 32);
                    dim3 gridSize((out_h * out_w + blockSize.x - 1) / blockSize.x,
                                  (nobs * out_channels + blockSize.y - 1) / blockSize.y);

                    convolve_kernel<<<gridSize, blockSize>>>(nobs, in_channels, out_channels, in_h, in_w, k_h, k_w, out_h, out_w, d_in, d_w, d_out);
                    cudaDeviceSynchronize();

                    cudaMemcpy(layer->m_z.data(), d_out, out_numelem * sizeof(Scalar), cudaMemcpyDeviceToHost);

                    cudaFree(d_in);
                    cudaFree(d_w);
                    cudaFree(d_out);
                }
            public: 
                virtual ~GPUForwardStrategy() = default; 
                virtual void forward(const Matrix& prev_layer_data, Convolutional<Activation>* layer) override {
                    // Each column is an observation
                    const int nobs = prev_layer_data.cols(); // Number of images in the batch 
                    // Linear term, z = conv(in, w) + b
                    layer->m_z.resize(layer->m_out_size, nobs);
                    
                    // Convolution
                    auto start = std::chrono::high_resolution_clock::now();
                    convolve(prev_layer_data, layer);
                    auto end = std::chrono::high_resolution_clock::now();
                    layer->m_last_conv_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

                    // Add bias terms
                    int channel_start_row = 0;
                    const int channel_nelem = layer->m_dim.conv_rows * layer->m_dim.conv_cols;

                    for (int i = 0; i < layer->m_dim.out_channels; i++, channel_start_row += channel_nelem)
                    {
                        layer->m_z.block(channel_start_row, 0, channel_nelem, nobs).array() += layer->m_bias[i];
                    }

                    // Apply activation function
                    layer->m_a.resize(layer->m_out_size, nobs);
                    Activation::activate(layer->m_z, layer->m_a);
                }
        };

        private: 
            std::unique_ptr<ForwardStrategy> m_forward_strategy;
        public: 
            double m_last_conv_time_ms;
            double get_last_conv_time() const { return m_last_conv_time_ms; }

        void set_strategy(ForwardStrategy* strategy) {
            m_forward_strategy.reset(strategy);
        }

        /// Constructor
        ///
        /// \param in_width      Width of the input image in each channel.
        /// \param in_height     Height of the input image in each channel.
        /// \param in_channels   Number of input channels.
        /// \param out_channels  Number of output channels.
        /// \param window_width  Width of the filter.
        /// \param window_height Height of the filter.
        ///
        Convolutional(const int in_width, const int in_height,
                      const int in_channels, const int out_channels,
                      const int window_width, const int window_height) :
            Layer(in_width * in_height * in_channels,
                  (in_width - window_width + 1) * (in_height - window_height + 1) * out_channels),
            m_dim(in_channels, out_channels, in_height, in_width, window_height,
                  window_width)
        {
            m_forward_strategy = std::make_unique<MiniDNNForwardStrategy>();
        }

        void init(const Scalar& mu, const Scalar& sigma, RNG& rng)
        {
            // Set data dimension
            init();
            // Random initialization of filter parameters
            const int filter_data_size = m_dim.in_channels * m_dim.out_channels *
                                         m_dim.filter_rows * m_dim.filter_cols;
            internal::set_normal_random(m_filter_data.data(), filter_data_size, rng, mu,
                                        sigma);
            // Bias term
            internal::set_normal_random(m_bias.data(), m_dim.out_channels, rng, mu, sigma);
        }

        void init()
        {
            // Set parameter dimension
            const int filter_data_size = m_dim.in_channels * m_dim.out_channels *
                                         m_dim.filter_rows * m_dim.filter_cols;
            // Filter parameters
            m_filter_data.resize(filter_data_size);
            m_df_data.resize(filter_data_size);
            // Bias term
            m_bias.resize(m_dim.out_channels);
            m_db.resize(m_dim.out_channels);
        }

        // http://cs231n.github.io/convolutional-networks/
        void forward(const Matrix& prev_layer_data)
        {
            m_forward_strategy->forward(prev_layer_data, this);
        }

        const Matrix& output() const
        {
            return m_a;
        }

        // prev_layer_data: in_size x nobs
        // next_layer_data: out_size x nobs
        // https://grzegorzgwardys.wordpress.com/2016/04/22/8/
        void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data)
        {
            const int nobs = prev_layer_data.cols();
            // After forward stage, m_z contains z = conv(in, w) + b
            // Now we need to calculate d(L) / d(z) = [d(a) / d(z)] * [d(L) / d(a)]
            // d(L) / d(a) is computed in the next layer, contained in next_layer_data
            // The Jacobian matrix J = d(a) / d(z) is determined by the activation function
            Matrix& dLz = m_z;
            Activation::apply_jacobian(m_z, m_a, next_layer_data, dLz);
            // z_j = sum_i(conv(in_i, w_ij)) + b_j
            //
            // d(z_k) / d(w_ij) = 0, if k != j
            // d(L) / d(w_ij) = [d(z_j) / d(w_ij)] * [d(L) / d(z_j)] = sum_i{ [d(z_j) / d(w_ij)] * [d(L) / d(z_j)] }
            // = sum_i(conv(in_i, d(L) / d(z_j)))
            //
            // z_j is an image (matrix), b_j is a scalar
            // d(z_j) / d(b_j) = a matrix of the same size of d(z_j) filled with 1
            // d(L) / d(b_j) = (d(L) / d(z_j)).sum()
            //
            // d(z_j) / d(in_i) = conv_full_op(w_ij_rotate)
            // d(L) / d(in_i) = sum_j((d(z_j) / d(in_i)) * (d(L) / d(z_j))) = sum_j(conv_full(d(L) / d(z_j), w_ij_rotate))
            // Derivative for weights
            internal::ConvDims back_conv_dim(nobs, m_dim.out_channels, m_dim.channel_rows,
                                             m_dim.channel_cols,
                                             m_dim.conv_rows, m_dim.conv_cols);
            internal::convolve_valid(back_conv_dim, prev_layer_data.data(), false,
                                     m_dim.in_channels,
                                     dLz.data(), m_df_data.data()
                                    );
            m_df_data /= nobs;
            // Derivative for bias
            // Aggregate d(L) / d(z) in each output channel
            ConstAlignedMapMat dLz_by_channel(dLz.data(), m_dim.conv_rows * m_dim.conv_cols,
                                              m_dim.out_channels * nobs);
            Vector dLb = dLz_by_channel.colwise().sum();
            // Average over observations
            ConstAlignedMapMat dLb_by_obs(dLb.data(), m_dim.out_channels, nobs);
            m_db.noalias() = dLb_by_obs.rowwise().mean();
            // Compute d(L) / d_in = conv_full(d(L) / d(z), w_rotate)
            m_din.resize(this->m_in_size, nobs);
            internal::ConvDims conv_full_dim(m_dim.out_channels, m_dim.in_channels,
                                             m_dim.conv_rows, m_dim.conv_cols, m_dim.filter_rows, m_dim.filter_cols);
            internal::convolve_full(conv_full_dim, dLz.data(), nobs,
                                    m_filter_data.data(), m_din.data()
                                   );
        }

        const Matrix& backprop_data() const
        {
            return m_din;
        }

        void update(Optimizer& opt)
        {
            ConstAlignedMapVec dw(m_df_data.data(), m_df_data.size());
            ConstAlignedMapVec db(m_db.data(), m_db.size());
            AlignedMapVec      w(m_filter_data.data(), m_filter_data.size());
            AlignedMapVec      b(m_bias.data(), m_bias.size());
            opt.update(dw, w);
            opt.update(db, b);
        }

        std::vector<Scalar> get_parameters() const
        {
            std::vector<Scalar> res(m_filter_data.size() + m_bias.size());
            // Copy the data of filters and bias to a long vector
            std::copy(m_filter_data.data(), m_filter_data.data() + m_filter_data.size(),
                      res.begin());
            std::copy(m_bias.data(), m_bias.data() + m_bias.size(),
                      res.begin() + m_filter_data.size());
            return res;
        }

        void set_parameters(const std::vector<Scalar>& param)
        {
            if (static_cast<int>(param.size()) != m_filter_data.size() + m_bias.size())
            {
                throw std::invalid_argument("[class Convolutional]: Parameter size does not match");
            }

            std::copy(param.begin(), param.begin() + m_filter_data.size(),
                      m_filter_data.data());
            std::copy(param.begin() + m_filter_data.size(), param.end(), m_bias.data());
        }

        std::vector<Scalar> get_derivatives() const
        {
            std::vector<Scalar> res(m_df_data.size() + m_db.size());
            // Copy the data of filters and bias to a long vector
            std::copy(m_df_data.data(), m_df_data.data() + m_df_data.size(), res.begin());
            std::copy(m_db.data(), m_db.data() + m_db.size(),
                      res.begin() + m_df_data.size());
            return res;
        }

        std::string layer_type() const
        {
            return "Convolutional";
        }

        std::string activation_type() const
        {
            return Activation::return_type();
        }

        void fill_meta_info(MetaInfo& map, int index) const
        {
            std::string ind = internal::to_string(index);
            map.insert(std::make_pair("Layer" + ind, internal::layer_id(layer_type())));
            map.insert(std::make_pair("Activation" + ind, internal::activation_id(activation_type())));
            map.insert(std::make_pair("in_channels" + ind, m_dim.in_channels));
            map.insert(std::make_pair("out_channels" + ind, m_dim.out_channels));
            map.insert(std::make_pair("in_height" + ind, m_dim.channel_rows));
            map.insert(std::make_pair("in_width" + ind, m_dim.channel_cols));
            map.insert(std::make_pair("window_width" + ind, m_dim.filter_cols));
            map.insert(std::make_pair("window_height" + ind, m_dim.filter_rows));
        }
};


} // namespace MiniDNN


#endif /* LAYER_CONVOLUTIONAL_H_ */

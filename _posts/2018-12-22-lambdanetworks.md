---
layout: distill
title: Lambda Networks
description: Modeling long-range interaction without attention.
date: 2020-10-26
comments: True

authors:
  - name: Duy V. Huynh
    url: "vndee.github.io"
    affiliations:
      name: vndee.github.io

bibliography: 2020-10-26-lambdanetworks.bib

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }

---

**Lambda Network**<d-cite key="anonymous2021lambdanetworks"/> là kiến trúc Neural Network mới nhất (tính tới thời điểm viết bài <d-footnote>Paper của Lambda Network đang trong quá trình double-blind open peer review cho hội nghị ICLR 2021.</d-footnote>) đạt SOTA trên tập dữ liệu ImageNet. Các tác giả đề xuất **lambda layer** có khả năng trích xuất được các thông tin phụ thuộc xa (long-range dependency) trong nhiều trường hợp như global, local và mask context. Những thông tin lambda layer có thể trích xuất được bao gồm cả tương quan về tính chất lẫn tương quan về vị trí <d-footnote>Content-based và position-based.</d-footnote>. Điểm đặc biệt là mô hình này gọn nhẹ và nhanh hơn nhiều lần so với các mô hình **self-attention** và **CNN** thông thường. Vậy có điều gì đặc biệt trong mô hình này so với các kiến trúc khác hiện nay.

## Attention Networks

Trước khi tìm hiểu về **Lambda Network**, trước tiên chúng ta hãy nhìn lại một chút về *attention*, xương sống của cuộc cách mạng **Transformer**. Khai thác thông tin phụ thuộc xa là một trong những vấn đề nan giải của Machine Learning nói chung và Deep Learning nói riêng. Có thể thấy rằng những mô hình truyền thống như RNN, CNN đều cố gắng để khác thác sự tương quan giữa các thành phần của dữ liệu. Điều khiến RNN và các biến thể (LSTM, GRU,..) trở thành tiêu chuẩn cho các mô hình xử lý ngôn ngữ tự nhiên, còn CNN thường dùng cho thị giác máy tính là vì đặc trưng dữ liệu thích hợp với mô hình nào hơn mà thôi. Trong thực tế, CNN vẫn có thể dùng cho các bài toán về ngôn ngữ rất tốt, đặc biệt là các mô hình TextCNN<d-cite key="kim2014"/>, CharCNN<d-cite key="zhang2015character"/>. Mặc khác, ở chiều ngược lại sẽ là khá khó khăn về vấn đề tính toán nếu xem mỗi bức ảnh là một chuỗi các pixel liền kề nhau để áp dụng mô cho những mô hình RNN truyền thống. Mặc khác, học biểu diễn thông qua mạng CNN có vẻ có mức độ phân cấp tốt hơn khi qua mỗi tầng thông tin được tổng hợp sẽ đi từ mức độ local context tới global context. Trong khi đó, các mạng hồi quy tổng hợp thông tin theo một trục nhất định của miền dữ liệu (thường là time-step), kiểu mô hình đặc biệt hiệu quả cho các bài toán *sequence-to-sequence*. Tuy nhiên có hai vấn đề rất lớn đối với các mạng hồi quy đó là vanishing gradient<d-footnote>Dù có nhiều cải tiến nhưng cũng không thể giải quyết triệt để vấn đề này.</d-footnote> và hiệu quả tính toán<d-footnote>Do tính chất của mạng nên khó có thể tận dụng tối đa khả năng tính toán song song.</d-footnote>.

<p align="center">
  <img src="/assets/img/00.png" alt="Place holder image"/>
</p>

Trong bối cảnh đó, Transformer<d-cite key="vaswani2017attention"/> ra đời với mục tiêu giải quyết hai vấn đề nêu trên của mạng hồi quy và tạo một bước đột phá mới trong lĩnh vực xử lý ngôn ngữ tự nhiên. Kiến trúc Transformer được hình thành bởi nhiều lớp (multi-head) self-attention layer theo sau bởi *Point-wise Feed Forward Network*. Trong mô hình mạng dạng này, thông tin về vị trí và thứ tự của từng thành phần dữ liệu thường được tích hợp vào input trước khi đưa qua các layer biến đổi dữ liệu. Do đó, trong mô hình Transformer người ta thường sử dụng *positional encoding* để tích hợp vào trong input. Có rất nhiều công thức attention nhưng công thức được sử dụng trong Transformer đó là **Scale Dot-Product**. Công thức này lấy cảm hứng từ phép truy vấn cơ sở dữ liệu nên được mô tả như sau:

$$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

Trong đó:
- $Q, K$: Lần lượt là vector query và vector key ($\|Q\|=\|K\|=d_k$). 
- $V$: Vector value ($\|V\|=d_v$)

Giả sử ta có vector embedding $x$ đã bao gồm thông tin về vị trí (positional encoding), để có được 3 vector $Q, K, V$ ta cần lấy $x$ nhân lần lượt với 3 ma trận $W^Q, W^K, W^V$. Đây chính là 3 ma trận tham số cần học được của Self-Attention trong mô hình Transformer. Vì vậy công thức trên có thể viết lại thành:

$$\text{Attention}(x) = \text{softmax}(\frac{xW^Q(xW^K)^T}{\sqrt{d_k}})xW^V$$

## Lambda Networks

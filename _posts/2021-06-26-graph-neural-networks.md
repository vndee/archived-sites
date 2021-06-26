---
layout: distill
title: Graph Neural Networks
description: Graph Neural Networks toàn cảnh
date: 2021-06-26
comments: True

authors:
  - name: Duy V. Huynh
    url: "vndee.github.io"
    affiliations:
      name: vndee.github.io

bibliography: 2021-06-26-graph-neural-networks.bib

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

Trong suốt thời gian qua, chúng ta đã chứng kiến sự phát triển như vũ bão của học sâu cùng hàng loạt ứng dụng trong lĩnh vực xử lý ngôn ngữ tự nhiên và thị giác máy tính. Các mô hình học sâu nổi tiếng như CNN, LSTM, Transformer,.. đạt được nhiều thành tựu trong nhiều bài toán với các dạng dữ liệu như ảnh và dữ liệu dạng văn bản. Qua đó mở ra những tham vọng có thể áp dụng học sâu cho các hình thức dữ liệu khác đa dạng hơn. Trong phạm vi bài viết này, ta sẽ đề cập đến hình thức dữ liệu đồ thị, cùng tìm hiểu xem tại sao chúng ta cần giải quyết các bài toán trên đồ thị và học sâu đã được áp dụng như thế nào trên dạng dữ liệu này.

## Đồ thị là gì?

Đồ thị mà chúng ta nhắc đến ở đây chính là đồ thị trong lý thuyết đồ thị (graph theory). Lý thuyết đồ thị là một ngành tương đối lớn trong toán học và ứng dụng của nó xuất hiện trong nhiều ngành khoa học khác nhau. Đặc biệt trong khoa học máy tính thì lý thuyết đồ thị là một nền tảng quan trọng. Mạng internet toàn cầu chính là một đồ thị siêu lớn với hàng tỉ thiết bị được kết nối thông qua mạng viễn thông. Và những bài toán xuất hiện trong các đồ thị như vậy chính là đối tượng nghiên cứu chính của ngành này.

Lục lại lịch sử một chút, bài toán đầu tiên được phát biểu chính thức dưới dạng một đồ thị có lẽ là bài toán ["7 cây cầu ở Königsberg"](https://en.wikipedia.org/wiki/Seven_Bridges_of_K%C3%B6nigsberg) của nhà toán học lừng danh [Leonhard Euler](https://en.wikipedia.org/wiki/Leonhard_Euler) vào năm 1736<d-cite key="book1"/>. Một cách ngắn gọn, bài toán này có thể mô tả như sau: 

> Thành phố Königsberg bị chia cách thành nhiều phần bởi một dòng song tên là [Pregel](https://en.wikipedia.org/wiki/Pregolya). Tất cả các phần của thành phố được kết nối với nhau thông qua 7 cây cầu. Bài toán được đặt ra là tìm đường đi qua tất cả các phần của thành phố sao cho mỗi cây cầu chỉ được đi qua duy nhất một lần. - Theo [Wikipedia](https://en.wikipedia.org/wiki/Seven_Bridges_of_K%C3%B6nigsberg).

<div class="l-body">
	<p align="center">
	  	<img src="http://i.imgur.com/fVNX7e3.png" width="50%" height="50%"/>
	</p>
</div>

<div class="l-body">
	<p align="center">
	  	<img src="http://jalammar.github.io/images/t/Transformer_decoder.png" width="100%" height="100%"/>
	</p>
</div>

<!-- ![](https://i.imgur.com/fVNX7e3.png) -->
---
layout: distill
title: Minh họa mô hình Transformer
description: Bài dịch từ bài viết của tác giả Jay Alammar.
date: 2020-10-31
comments: True

authors:
  - name: Jay Alammar
    url: "http://jalammar.github.io/"
    affiliations:
      name: http://jalammar.github.io/

bibliography: 2020-10-31-the-illustrated-transformer.bib

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

Trong một bài viết trước đây, chúng ta đã tìm hiểu về [cơ chế attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) - một phương pháp phổ biến trong các mô hình Deep Learning hiện đại. Attention là khái niệm đã giúp cải thiện hiệu suất của các ứng dụng dịch máy sử dụng mạng nơ-ron. Trong bài viết này, chúng ta sẽ tìm hiểu về mô hình **Transformer** - một mô hình sử dụng attention để tăng tốc với các mô hình có thể được huấn luyện. Transformer vượt trội hoàn toàn nếu so với mô hình dịch máy sử dụng mạng nơ-ron của Google <d-footnote>Google Neural Machine Translation.</d-footnote> trên nhiều tác vụ. Điểm vượt trội lớn nhất của mô hình này chính là khả năng song song hóa. Thực tế, Google Cloud khuyến khích chúng ta nên sử dụng Transformer như một ví dụ tham khảo cho việc sử dụng [Cloud TPU](https://cloud.google.com/tpu/) của họ. Vì vậy, hãy thử tách mô hình ra thành từng phần và xem nó hoạt động như thế nào.

Transformer được đề xuất trong công bố mang tên "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)" của A. Vasvawni cà các cộng sự <d-cite key="vaswani2017attention"/>. Phiên bản Transformer được hiện thực sử dụng Tensorflow là một phần của thư viện mã nguồn mở [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor). Harvard NLP Group cũng có một hướng dẫn và chú thích chi tiết về cách hiện thực Transformer bằng PyTorch <d-footnote>http://nlp.seas.harvard.edu/2018/04/03/attention.html</d-footnote>. Trong bài viết này, chúng ta sẽ cố gắng đơn giản hóa và trình bày từng khái niệm một để hi vọng nó dễ hiểu cho những người không có kiến thức quá chuyên sâu.

## Góc nhìn toàn cảnh

Hãy bắt đầu bằng cách nhìn vào mô hình Transformer như một black-box. Trong bài toán dịch máy, chúng ta mong muốn nhận dữ liệu đầu vào là một câu của ngôn ngữ nguồn và output kì vọng là bản dịch của nó trong ngôn ngữ đích.

<div class="l-page">
	<p align="center">
  		<img src="http://jalammar.github.io/images/t/the_transformer_3.png"/>
  	</p>
</div>

Đi sâu vào bên trong Optimus Prime <d-footnote>Nhân vật trong phim điện ảnh cùng Transformer.</d-footnote>, chúng ta thấy một bộ mã hóa (encoding component), một bộ giãi mã (decoding component) và các liên kết giữa chúng. 
 

<div class="l-body">
	<p align="center">
	  	<img src="http://jalammar.github.io/images/t/The_transformer_encoders_decoders.png"/>
	</p>
</div>

Bộ mã hóa (encoding component) là một khối gồm các lớp encoder được xếp liên tiếp nhau (bài báo gốc dùng 6 encoder xếp chồng lên nhau - không có gì huyền bí về con số 6 này, chúng ta hoàn toàn có thể thử nghiệm với nhiều cách sắp xếp và số tầng khác nhau.

<div class="l-body">
	<p align="center">
	  	<img src="http://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png" width="100%" height="100%"/>
	</p>
</div>

Các encoder đều giống nhau về cấu trúc (chúng không dùng chung bộ tham số). Mỗi lớp encoder được cấu tạo gồm 2 thành phần nhỏ hơn như sau:

<div class="l-body">
	<p align="center">
	  	<img src="http://jalammar.github.io/images/t/Transformer_encoder.png" width="100%" height="100%"/>
	</p>
</div>

Dữ liệu đâu vào của encoder trước tiên sẽ đi qua lớp self-attention - một lớp giúp encoder xem xét sự liên quan của các từ khác trong câu với từ cụ thể mà nó mã hóa. Chúng ta sẽ tiếp tục đi sâu hơn vào self-attention.

Sau khi đi qua lớp self-attention, dữ liệu tiếp tục được đưa qua một mạng nơ-ron truyền thẳng (Feed-Forward Neural Network). Mạng FFNN giống hệt nhau được áp dụng độc lập cho từng vị trí.

Bộ giãi mã (decoding component) có cả hai lớp nói trên, tuy nhiên ở giữa chúng có một lớp attention nữa để giúp decoder tập trung vào những phần liên quan của câu đầu vào (giống như cách attention được sử dụng trong [mô hình seq2seq](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)).

<div class="l-body">
	<p align="center">
	  	<img src="http://jalammar.github.io/images/t/Transformer_decoder.png" width="100%" height="100%"/>
	</p>
</div>

## Mang Tensor vào trong bức tranh

Bây giờ chúng ta đã biết được những thành phần cở bản của mô hình, hãy bắt đầu tìm hiểu những vector/tensor trong mô hình và chúng hoạt động, tương tác như thế nào để biến đổi dữ liệu.

Như một cách tiếp cận tiêu chuẩn cho mọi bài toán NLP, chúng ta bắt đầu bằng việc biến đổi từng từ trong câu đầu vào thành vector sử dụng các [thuật toán embedding](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca).


<div class="l-body">
	<p align="center">
	  	<img src="http://jalammar.github.io/images/t/embeddings.png" width="100%" height="100%"/>
	  	<figcaption align="center">Mỗi từ được mã hóa thành một vector 512 chiều. Chúng ta sẽ thể hiện các vector này bằng những box như trên.</figcaption>
	</p>
</div>

Lớp embedding này chỉ xuất hiện ở của encoder phía dưới cùng. Một cách tổng quát, tất cả các encoder đều nhận vào danh sách các vector 512 chiều. Tuy nhiên chỉ có đầu của layer đầu tiên (dưới cùng) là word embedding, còn các layer ở tầng cao hơn thì đầu vào của nó chính là đầu ra của layer trước đó. Kích thước danh sách các vector là một siêu tham số được cài đặt sẵn, về cơ bản nó thường là độ dài của câu dài nhất trong tập dữ liệu huấn luyện.

Sau khi có được vector word embedding cho từng từ trong câu. Mỗi vector này sẽ được đi qua lần lượt hay layer nhỏ hơn bên trong encoder là self-attention và feed-forward neural network.

<div class="l-body">
	<p align="center">
	  	<img src="http://jalammar.github.io/images/t/encoder_with_tensors.png" width="100%" height="100%"/>

	</p>
</div>

Ở đây chúng ta dễ thấy một tinh chất của Transformer, mỗi từ tại mỗi vị trí sẽ đi một đường của riêng mình bên trong encoder. Có sự phụ thuộc về thông tin giữa các đường đi này trong lớp self-attention. Còn tại lớp FFNN thì không có sự phụ thuộc này. Tuy nhiên, nhờ vậy mà sẽ có rất nhiều đường đi có thể thực hiện song song khi đi qua lớp FFNN.

Tiếp theo chúng ta cùng xem xét một ví dụ cụ thể để biết điều gì thật sự diễn ra khi thông tin đi qua hai lớp con của encoder.

## Mã hóa

Như đã đề cập, encoder nhận một danh sách các vector là dữ liệu đầu vào. Encoder xử lý các vector này bằng cách đẩy chúng vào một lớp self-attention, sau đó cho đi qua lớp FFNN, cuối cùng là đưa vector output đến tầng encoder tiếp theo.


<div class="l-body">
	<p align="center">
	  	<img src="http://jalammar.github.io/images/t/encoder_with_tensors_2.png" width="100%" height="100%"/>
		<figcaption align="center">Mỗi từ tại mỗi vị trí đi qua lớp self-attention. Sau đó chúng lần lượt đi qua lớp FFNN.</figcaption>
	</p>
</div>

## Self-Attention từ trên cao

Đừng nghĩ rằng tôi viết "self-attention" như một khái niệm tất cả mọi người đều phải quen thuộc. Cá nhân tôi chưa bao giờ biết đến khái niệm này cho đến khi đọc bài báo "Attention Is All You Need". Chúng ta cùng tóm tắt ngắn gọn cách mà nó hoạt động.

Giả sử câu sau đây là câu đầu vào mà chúng ta cần dịch:

`The animal didn't cross the street because it was too tired`

`It` trong câu trên chỉ đến đối tượng nào? Nó ngụ ý cho `street` hay `animal`? Đó là một câu hỏi đơn giản dành cho con người, nhưng không hề đơn giản với thuật toán.

Khi mô hình sử dụng **self-attention** đang trong quá trình xử lí đối với từ `it`, **self-attention** cho phép mô hình tự động quan tâm đến mối liên kết giữa `it` và `animal`. Khi mô hình xử lí từng từ (từng vị trí trong chuỗi đầu vào), **self-attention** sẽ quan sát những từ ở những vị trí khác trong câu để tìm những sự tương quan giúp mô hình mã hóa từ đang xử lí tốt hơn.

Nếu bạn quen thuộc với RNN, hãy nghĩ về cách RNN duy trì *hidden state* để mô hình có thể kết hợp vector biểu diễn của các từ đã được xử lý trước đó với từ đang được xử lý tại bước hiện tại. **Self-attention** chính là phương pháp mà **Transformer** dùng để đưa nhưng thông tin của các từ khác vào quá trình xử lý từ hiện tại thông qua độ tương quan giữa chúng.

<div class="l-body">
	<p align="center">
	  	<img src="https://jalammar.github.io/images/t/transformer_self-attention_visualization.png"/>
		<figcaption align="center">Khi chúng ta mã hóa từ "it" tại encoder thứ 5 (encoder trên cùng), một sự chú ý lớn của cơ chế attention được dành cho từ "The Animal" và đóng góp một phần vector biểu diễn của nó vào vector biễu diễn của "it".</figcaption>
	</p>
</div>

Để rõ hơn bạn hãy xem qua [notebook của Tensor2Tensor](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb), nơi bạn có thể load mô hình Transformer và sử dụng hình ảnh trực quan ở trên.

## Chi tiết về Self-Attention

Chúng ta hãy cùng xem làm sao để tính **self-attention** sử dụng các vector, và tìm hiểu cách chúng được hiện thực bằng các ma trận.

**Bước đầu tiên** trong việc tính **self-attention** là tạo ra 3 vector từ vector đầu vào của encoder (trong trường hợp này là vector word embedding). Với mỗi từ, chúng ta tạo một vector **Query**, một vector **Key** và một vector **Value**. Những vector này được tạo bởi bằng cách nhân vector word embedding với 3 ma trận trọng số mà chúng ta sẽ huấn luyện.

Chú ý rằng 3 vector được tạo ra có số chiều nhỏ hơn vector embedding. Số chiều của chúng là 64, trong khi đó vector embedding và input/output của encoder có số chiều là 512. Mặc khác, chúng không nhất thiết phải nhỏ hơn, đây chỉ là một lựa chọn của mô hình để khiến công việc tính toán multi-head attention (gần như) là cố định.

<div class="l-body">
	<p align="center">
	  	<img src="https://jalammar.github.io/images/t/transformer_self_attention_vectors.png" width="100%" height="100%"/>
		<figcaption align="center">Nhân <b>x1</b> với <b>WQ</b> ta được <b>q1</b> - vector "query" tương ứng của từ. Cuối cùng chúng ta tạo ra 3 vector "query", "key", "value" của mỗi từ trong câu đầu vào.</figcaption>
	</p>
</div>

Vậy ý nghĩa của "query", "key", "value" là gì?

Những khái niệm trừu tượng này rất hữu ích trong việc tính toán và suy nghĩ về attention. Khi bạn đọc cách tính attention bên dưới, bạn sẽ hiểu ra khá nhiều điều về vai trò của 3 vector này.

**Bước thứ hai** đó là tính toán self-attention score. Giả sử chúng ta đang tính self-attention score cho từ đầu tiên trong ví dụ này `"Thinking"`. Chúng ta cần tính score của các từ khác trong câu với từ `"Thinking"`. Score này quy định độ tương quan của từ `"Thinking"` đến những từ khác khi mã hóa. Score được tính bằng cách tính tích vô hướng (dot product) giữa vector query và vector key. Vì vậy để tính self-attention score tại vị trí số 1, chỉ cần tính tích vô hướng của **q1** và **k1**, score tại vị trí số 2 sẽ là **q1** và **k2**.

<div class="l-body">
	<p align="center">
	  	<img src="https://jalammar.github.io/images/t/transformer_self_attention_score.png" width="100%" height="100%"/>
	</p>
</div>

**Bước ba thứ ba** và **bước thứ tư** là chia các score tính được cho 8 (đây là căn bậc hai số chiều của vector key được dùng trong bài báo - tức 64. Chia cho 8 giúp chúng ta có được gradient ổn định hơn. Đây hoàn toàn có thể là một con số khác tuy nhiên 8 là số mặc định), sau đó cho kết quả này đi qua một hàm softmax. Softmax giúp chuẩn hóa các score thành các số dương và có tổng bằng 1.

<div class="l-body">
	<p align="center">
	  	<img src="https://jalammar.github.io/images/t/self-attention_softmax.png" width="100%" height="100%"/>
	</p>
</div>

Điểm softmax quyết định độ quan trọng của từ tại ví trí này so với từ đang được xử lý. Dễ dàng nhận thấy, nếu nhân vector query và vector key tại cùng một ví trí thì nó sẽ cho ra điểm softmax cao nhất, nhưng đôi khi nó cũng hữu ích để chú ý đến những từ khác có liên quan với từ hiện tại.

**Bước thứ năm** là nhân mỗi vector value với softmax score (để chuẩn bị cộng chúng lại). Ý tưởng ở đây là giữ lại giá trị value của các từ mà chúng ta tập trung vào, còn hạ thấp giá trị của những từ không liên quan đi (bằng cách nhân chúng với 0.001 chẳng hạn).

**Bước thứ sáu** là tính tổng của các vector value này. Đây chính là kết quả của self-attention layer tại mỗi vị trí (cho từ đầu tiên).

<div class="l-body">
	<p align="center">
	  	<img src="https://jalammar.github.io/images/t/self-attention-output.png" width="100%" height="100%"/>
	</p>
</div>

Quá trình tính toán self-attention cho kết quả là một vector, sau đó tiêp tục được đưa qua Feed Forward Neural Network. Trong thực tế, quá trình tính toán này được thực hiện bằng các phép toán ma trận với mục tiêu xử lý nhanh hơn. Vì vậy, bây giờ chúng ta đã thấy được ý nghĩa của việc tính toán self-attention ở cấp độ từ.

## Tính self-attention bằng phép toán ma trận

**Bước đầu tiên** là tính 3 ma trận query, key và value. Điều đó được thực hiện bằng cách stack các vector embedding thành một ma trận đầu vào X và nhân lần lượt với các ma trận trọng số (**WQ, WK, WV**).

<div class="l-body">
	<p align="center">
	  	<img src="https://jalammar.github.io/images/t/self-attention-matrix-calculation.png" width="100%" height="100%"/>
		<figcaption align="center">Mỗi hàng của ma trận X tương ứng với vector embedding của 1 từ trong câu đầu vào. Chúng ta lại thấy sự khác biệt về kích thước của các vector embedding (512 hay 4 ô như trong hình) và các vector q/k/v (64 hay 3 ô như trong hình).</figcaption>
	</p>
</div>

**Cuối cùng**, vì chúng ta đang làm việc với ma trận, chúng ta có thể gộp các bước từ bước hai đến bước sáu thông qua một công thức như sau:

<div class="l-body">
	<p align="center">
	  	<img src="https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png" width="100%" height="100%"/>
		<figcaption align="center">Tính self-attention bằng phép toán ma trận.</figcaption>
	</p>
</div>

## Quái vật nhiều đầu

Bài báo gốc đã cải tiến mô hình sử dụng self-attention bằng cách thêm một khái niệm gọi là "multi-head" (nhiều đầu) attention. Nó giúp cải thiện hiệu suất của mô hình theo hai cách:

- Gia tăng khả năng của mô hình để tập trung sự chú ý vào nhiều vị trí khác nhau. Đúng vậy, trong ví dụ trên, z1 chứa rất ít thông tin của các vector mã hóa khác, nhưng nó bị chi phối bởi chính từ đó. Nó sẽ rất hữu ích khi chúng ta muốn dịch câu `The animal didn’t cross the street because it was too tired`, mà chúng ta biết được từ `it` chỉ đến từ nào trong câu.
- Kết quả của lớp attention là vector biểu diễn trong không gian kết hợp của nhiều không gian biểu diễn nhỏ hơn. Như chúng ta thây dưới đây, với multi-head attention chúng ta không chỉ có một mà nhiều mâ trận trọng số Query/Key/Value khác nhau (Mô hình Transformer dùng 8 attention head, do đó cuối cùng chúng ta có được 8 bộ ma trận cho mỗi encoder/decoder). Tất cả chúng đều được khởi tọa ngẫu nhiên. Sau đó sẽ được update trong quá trình huấn luyện để biến đổi vector đầu vào thành vector trong nhiều không gian biểu diễn con.

<div class="l-body">
	<p align="center">
	  	<img src="https://jalammar.github.io/images/t/transformer_attention_heads_qkv.png" width="100%" height="100%"/>
		<figcaption align="center">Với multi-head attention, chúng ta duy trì nhiều bộ Q/K/V khác nhau cho mỗi head. Như đã làm trước đó, chúng ta nhân X với ma trận WQ/WK/WV để được ma trận Q/K/V.</figcaption>
	</p>
</div>


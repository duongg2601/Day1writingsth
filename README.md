# Day1writingsth

Ngày nay, trí tuệ nhân tạo đang được sử dụng mọi nơi. Và nhiều ứng dụng đột phá đến từ Học Máy, một mảng con của AI.

Trong Học máy, một mảng gọi là Học sâu(Deep Learning) thì khá mới, là hệ thống AI thực sự hiệu quả.
Nhưng thường thì hệ thống AI tạo ra từ Học sâu khá hẹp và tập trung. Chúng vượt trội hơn con người trong một lĩnh vực đặc biệt nơi mà nó được tạo ra.
Bởi lẽ đó, rất nhiều các cải tiến mới của AI đến tử những hệ thống đặc biệt hoặc một tổ hợp của các hệ thống làm việc cùng nhau.
Một trong những vấn đề lớn nhất trong lĩnh vực mô hình Học sâu là nó thiếu khả năng diễn giải(interpretability). Interpretability nghĩa là hiểu cách sự quyết định được tạo ra.
Đó là một vấn đề lớn mà có mảng riêng của nó, gọi là explainable AI. Đây là lĩnh vực trong AI mà tập trung vào việc khiến quyết định của mô hình AI có thể dễ hiểu hơn nhiều.

+Trí tuệ nhân tạo và sự trỗi dậy của Học sâu
  Học sâu trong AI là gì?
  Học sâu là một mảng con của AI. Nó sử dụng các mạng nơ-ron để xử lý những cấu trúc phức tạp(process complex patterns), 
như kiểu các chiến lược một đội bóng sử dụng để chiến thắng.
  Mạng nơ-ron càng lớn, thì khả năng nó làm được một điều lớn lao càng cao - như ChatGPT, nó sử dụng việc xử lý ngôn ngữ tự nhiên 
để trả lời câu hỏi và tương tác với người dùng.
  Để thực sự hiểu điều cơ bản về mạng nơ-ron - điểm chung mà mỗi một mô hình AI đều có để cho phép nó hoạt động - ta cần hiểu 
các lớp kích hoạt(activation layers).

![Simple neural network](https://github.com/user-attachments/assets/1d56e22c-beb4-4a03-858d-8fbec076133a)

  Cốt lõi của Học sâu là huấn luyện mạng nơ-ron. Có nghĩa là sử dụng dữ liệu để đạt được đúng giá trị ở mỗi nơ-ron, từ đó có thể dự đoán cái ta muốn. Mạng nơ-ron được tạo từ các nơ-ron được tổ chức trong các lớp. Mỗi lớp phân tách các đặc tính riêng biệt từ dữ liệu. Cấu trúc phân lớp này cho phép mô hình Học sâu phân tích và diễn giải dữ liệu phức tạp.

+Vấn đề lớn trong Học sâu: Thiếu tính diễn giải
  Học sâu tạo cách mạng trong nhiều lĩnh vực bởi nó đạt nhiều kết quả to lớn với những nhiệm vụ rất phức tạp. Tuy vậy nó gặp vấn đề lớn trong việc thiếu tính diễn giải. Sự thật là khi mạng nơ-ron có thể hoạt động cực tốt, nhưng chúng ta không hiểu nội bộ trong mạng nơ-ron cách nó có thể đạt được kết quả đó. Hay nói cách khác, chúng ta biết nó làm tốt nhiệm vụ được giao nhưng không hiểu cách chi tiết nó làm.
  Thực sự rất quan trọng khi ta hiểu cách mô hình suy nghĩ trong các lĩnh vực như Y tế hay Lái xe tự hành(autonomous driving). Bởi khi hiểu cách mô hình suy nghĩ, ta có thể tự tin hơn về sự tin cậy của nó trong một số lĩnh vực quan trọng. Nghĩa là mô hình hoạt động ở các lĩnh vực mà có quy định chặt chẽ sẽ minh bạch hơn và xây dựng nhiều niềm tin hơn khi mà nó có thể được diễn giải. Mô hình mà cho phép khả năng diễn giải gọi là glass box models(hộp trong suốt). Ngược lại, mô hình không có khả năng đó gọi là black box models(hộp đen).

+Lời giải cho tính diễn giải: Glass Box Models
  Đây là mô hình học máy được thiết kế để dễ dàng được hiểu bởi con người. Nó cung cấp quan sát rõ ràng cách nó đưa ra quyết định. Tính trong suốt(minh bạch) này trong quá trình đưa quyết định là quan trọng trong việc tin tưởng, tuân thủ, và cải tiến. Bên dưới là code mẫu một mô hình AI dựa trên tập dữ liệu để dự đoán ung thư vú, nó đạt 97% độ chính xác. Ta cũng sẽ dự trên các đặc điểm của dữ liệu tìm ra điều quan trọng hơn trong việc dự đoán tỉ lệ ung thư.

+Black Box Models
  Đây là kiến trúc mạng nơ-ron khác biệt trong việc sử dụng các tập dự liệu. Vài ví dụ:  
  - CNN (Convolutional Neural Networks): thiết kế đặc biệt cho phân loại ảnh và thông giải.
  - RNN (Recurrent Neural Networks) and LSTM (Long Short Term Memory): sử dụng chủ yếu cho dữ liệu tuần tự - văn bản và chuỗi thời gian. Năm 2017, nó bị vượt qua bởi kiến trúc mạng nơ-ron gọi là transformers.
  - Transformer-based architectures: nó tạo ra cách mạng trong ngành AI vào 2017 bởi khả năng xử lý dữ liệu tuần tự hiệu quả hơn. RNN và LSTM có những hạn chế trong vấn đề này.
    
  Ngày nay, phần lớn mô hình mà xử lý văn bản là mô hình transformers-based. Ví dụ, ChatGPT, GPT viết tắt của Generative Pre-trained Transformer, đề cập đến kiến trúc mạng nơ-ron biến hình mà sinh ra văn bản. Tất cả các mô hình CNN, RNN, LSTM và Transformers là ví dụ của narrow AI (ý là từng mô hình AI sẽ làm tốt ở một mảng nhỏ đặc biệt). Vì vậy để đạt được general intelligence liên quan đến việc kết hợp những mô hình narrow AI để bắt chước hành vi của con người.

+Code mẫu Giải vấn đề với Explainable AI
  Code này, ta sẽ tạo mô hình dựa trên 30 đặc tính. Ta cũng sẽ tìm 5 đặc tính mà quan trọng hơn cả trong việc phát hiện ung thư vú, dựa trên tập dữ liệu đó. Ta sẽ dùng mô hình machine learning glass box gọi là Explainable Boosting Machine.

  Fullcode![image](https://github.com/user-attachments/assets/689505da-10c4-43cd-8c91-2b5623741ffb)

+Kết luận: KAN (Kolmogorov-Arnold Netwworks)
  Vì có explainable AI, ta có thể nghiên cứu dân số sử dụng phương pháp new data-driven.
  Thay vì chỉ sử dụng thống kê truyền thống, khảo sát, và phân tích dữ liệu thông thường, ta có thể mô tả kết luận chính xác hơn khi sử dụng thư viện chương trình AI và một cơ sở dữ liệu hoặc file Excel.
  Nhưng đó không phải cách duy nhất để xây dựng mô hình với explainable AI.
  Tháng 4 2024, bài báo tên KAN:Kolmogorov-Arnold Networks được xuất bản đã làm thay đổi lĩnh vực này nhiều hơn nữa. KANs hứa hẹn ngày một chính xác hơn và dễ hiểu hơn những mô hình truyền thống và hiệu năng tốt hơn.
  Nó cũng dễ dàng dể trực quan hóa và tương tác. 
  
  

  

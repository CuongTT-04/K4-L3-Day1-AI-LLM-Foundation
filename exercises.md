# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature = 0.0, câu trả lời mang tính tất định (deterministic), tập trung vào các sự thật phổ biến nhất (như hang Sơn Đoòng hoặc xuất khẩu cà phê/hạt điều) và lặp lại giống hệt nhau giữa các lần chạy. Khi tăng lên 0.5 - 1.0, văn phong trở nên đa dạng, phong phú từ ngữ và kể những khía cạnh mới lạ hơn. Khi lên tới 1.5, câu trả lời có độ ngẫu nhiên quá cao, từ ngữ bắt đầu trở nên lộn xộn, lan man và có nguy cơ xuất hiện cấu trúc câu bất thường.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature trong khoảng thấp từ 0.0 đến 0.3. Lý do là chatbot hỗ trợ khách hàng đòi hỏi tính chính xác cao, nội dung nhất quán và tuân thủ nghiêm ngặt chính sách/FAQ của doanh nghiệp; mức temperature thấp giúp hạn chế tối đa hiện tượng bịa đặt thông tin (hallucination) và tránh việc hai khách hàng nhận được các câu trả lời mâu thuẫn về cùng một chính sách.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Tổng lượng token đầu ra mỗi ngày là: 10.000 × 3 × 350 = 10.500.000 token (10.500K token). Chi phí output của GPT-4o là $105/ngày ($0.010/1K token), trong khi GPT-4o-mini chỉ tốn $6.3/ngày ($0.0006/1K token), nghĩa là GPT-4o đắt hơn khoảng 16.7 lần (chênh lệch gần $3.000/tháng). GPT-4o xứng đáng khi cần phân tích hợp đồng pháp lý phức tạp, tư vấn y tế hoặc sinh mã nguồn phần mềm đòi hỏi suy luận logic cao; ngược lại, GPT-4o-mini rất phù hợp cho tác vụ phân loại ý định người dùng (intent classification), tóm tắt văn bản ngắn hoặc trả lời FAQ đơn giản.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với vai giáo viên tiểu học, model sử dụng ngôn từ ngắn gọn, giọng điệu vui tươi và dùng hình ảnh ẩn dụ gần gũi như 'cuốn sổ ghi chép chung của cả lớp mà không ai được tự ý tẩy xóa'. Ngược lại, với vai chuyên gia tài chính, câu trả lời dài và trang trọng hơn, tập trung vào các thuật ngữ chuyên sâu như sổ cái phân tán (DLT), cơ chế đồng thuận (consensus), mật mã học phi tập trung và tính bất biến (immutability). System prompt hoạt động như một bộ khung chỉ dẫn, định hình vai diễn (persona), giọng điệu, mức độ trừu tượng và kho từ vựng mà model được phép sử dụng mà không cần phải tinh chỉnh (fine-tune) lại mô hình.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Thực nghiệm trên đoạn văn tiếng Việt 97 từ cho ra 224 token qua tiktoken (cl100k_base), trong khi ước lượng theo công thức (số từ / 0.75) chỉ là 129 token, chênh lệch thực tế cao hơn tới ~73%. Tiếng Việt tốn nhiều token hơn tiếng Anh vì thuật toán Byte Pair Encoding (BPE) của tokenizer được huấn luyện chủ yếu trên tập dữ liệu tiếng Anh; các từ tiếng Việt chứa nhiều ký tự có dấu thanh/dấu mũ (mã hóa đa byte UTF-8) và có cấu trúc từ ghép tách rời, khiến tokenizer thường phải xẻ nhỏ một từ thành 2–3 token (thậm chí từng byte riêng lẻ) thay vì gom thành một token duy nhất như từ vựng tiếng Anh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng hội thoại tương tác trực tiếp với người dùng cuối (như Chatbot AI, trợ lý ảo) vì nó giúp giảm thời gian phản hồi đầu tiên (Time to First Token - TTFT) từ vài giây xuống chỉ còn vài trăm mili-giây, mang lại cảm giác mượt mà và người dùng không phải nhìn màn hình loading trống rỗng. Ngược lại, non-streaming phù hợp hơn cho các tác vụ chạy ngầm (background jobs, batch processing), trích xuất dữ liệu, hoặc khi cần nhận về một chuỗi JSON có cấu trúc hoàn chỉnh để kiểm tra và parse cú pháp trước khi chuyển tiếp cho hệ thống khác xử lý.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> So với delay cố định, exponential backoff tăng dần thời gian chờ theo hàm số mũ (1s, 2s, 4s, 8s...), giúp giảm tần suất gửi yêu cầu và tạo khoảng nghỉ đủ dài cho server kịp giải phóng tài nguyên khi gặp quá tải (lỗi 429/503). Nếu hàng nghìn client cùng retry với thời gian cố định (ví dụ đúng 1 giây), sẽ xảy ra hiện tượng "cơn lũ retry" (Thundering Herd Problem): toàn bộ client sẽ đồng loạt dội lại request vào server tại cùng một thời điểm sau 1 giây, khiến server vừa ngóc đầu dậy lại tiếp tục sập và rơi vào vòng lặp quá tải vĩnh viễn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona là trợ giảng thân thiện cho sinh viên: "Bạn là trợ giảng thân thiện của khóa học AI LLM. Hãy trả lời ngắn gọn, súc tích bằng tiếng Việt, kèm ví dụ thực tế dễ hiểu và chỉ tập trung vào câu hỏi của sinh viên." Lựa chọn từ ngữ quan trọng: (1) Cụm từ "ngắn gọn, súc tích" giúp kiểm soát độ dài đầu ra, tiết kiệm token và giảm thiểu độ trễ phản hồi; (2) Chỉ định rõ "bằng tiếng Việt" đảm bảo tính nhất quán ngôn ngữ, ngăn model tự ý chuyển sang tiếng Anh khi gặp các thuật ngữ kỹ thuật chuyên sâu.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất hiện tại là cửa sổ ngữ cảnh bị giới hạn cứng ở 3 lượt hội thoại gần nhất (`history[-6:]`), khiến trợ lý nhanh chóng quên thông tin ban đầu khi cuộc trò chuyện kéo dài, đồng thời không có bộ nhớ dài hạn giữa các phiên làm việc. Cải thiện đề xuất: Tích hợp cơ chế "Tóm tắt ngữ cảnh tự động" (Context Summarization). Cách triển khai: Khi số tin nhắn vượt quá 6, thay vì cắt bỏ trực tiếp các tin nhắn cũ, ta gọi một model phụ siêu nhẹ (như `gpt-4o-mini`) để tóm tắt các điểm chính của cuộc trò chuyện cũ thành 1-2 câu ngắn, sau đó ghim bản tóm tắt này ngay dưới system prompt; cách này giúp giữ trọn vẹn ngữ cảnh của toàn bộ buổi trò chuyện mà không làm phình to số lượng token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026

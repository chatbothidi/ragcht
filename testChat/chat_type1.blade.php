<!-- chatbot.css -->
<link rel="stylesheet" href="/assets/css/chatbot.css">

<section id="chatbot-wrap" class="@empty($main_key) main @else sub @endempty">
    <div class="chatbot-btn">
        <div class="help-text" id="helpBubble">
            <span>호흡기 건강, 무엇이든 물어보세요! ☁️</span>
        </div>
        <button type="button" class="btn-chat-open" title="챗봇 열기">
            <span class="icon">☁️</span>
            <div class="dot"></div>
        </button>
    </div>

    <div class="chatbot-container" id="chatWindow">
        <div class="chatbot-header">
            <span class="icon">☁️</span>
            <h1 class="chatbot-tit">
                <strong>챗봇 (AI)</strong>
                <span>대한결핵 및 호흡기학회</span>
            </h1>
        </div>

        <div class="chatbot-contents" id="chatBody">
            <div class="chatbot-con bot">
                안녕하세요!<br> 대한결핵 및 호흡기학회 <b>챗봇</b>입니다.
                @if( thisAuth()->check() )
                <br>행사 참가 내역 등 학회 활동 내역은 왼쪽 하단의 사람 모양 버튼을 클릭하여 확인해 주세요.
                <br><br>메시지 입력 예시)) 올해 학회에서 진행하는 학술행사 알려줘
                @else
                <br>행사 참가 내역 등의 확인을 원하실 경우, 로그인 후 이용해주시기 바랍니다.
                <br><br>메시지 입력 예시) 올해 학회에서 진행하는 학술행사 알려줘
                @endif
            </div>
        </div>

        <div class="chatbot-footer">
            <div class="form-group">
                @if( thisAuth()->check() )
                    <button type="button" class="btn-chatmenu-open" id="menuBtn">
                        <span class="icon">
                            <span class="icon-head"></span>
                            <span class="icon-body"></span>
                        </span>
                    </button>
                @endif
                <input type="text" class="form-item chat-input" placeholder="메시지를 입력해주세요." id="chatInput">
                <button type="button" class="btn-send">➤</button>
            </div>

            <div class="chat-menu-wrap" id="menuDrawer">
                <button type="button" class="btn-menu" data-type="event">📑 사전등록 영수증 출력</button>
                <!-- <button type="button" class="btn-menu" data-type="fee">📑 회비 내역</button> -->
                <button type="button" class="btn-menu" data-type="active">📅 학술·교육활동 현황</button>
            </div>
        </div>

        <button type="button" class="btn-chat-close" onClick="toggleChat()" title="챗봇 닫기">×</button>
    </div>
</section>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

<script>
$(document).ready(function() {
    
    localStorage.clear();

    // 🔥 [추가] 마크다운 줄바꿈 옵션 활성화 (이게 없으면 줄글로 나옵니다)
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            breaks: true, // 엔터 한 번도 <br>로 변환
            gfm: true     // 깃허브 스타일 마크다운 사용
        });
    }

    // ==========================================
    // ★ 1. 페이지 로드 시 대화 내용 복구 & 초기화
    // ==========================================
    let chatHistory = JSON.parse(localStorage.getItem('chat_history')) || [];
    let currentThreadId = localStorage.getItem('chat_thread_id');

    // 저장된 대화가 있으면 화면에 뿌려줌
    if (chatHistory.length > 0) {
        chatHistory.forEach(function(item) {
            // 메시지 표시 함수 호출 (여기서 마크다운 변환됨)
            addMessage(item.content, item.sender);
        });
        // 스크롤 맨 아래로
        $('#chatBody').scrollTop($('#chatBody')[0].scrollHeight);
    }

    // ==========================================
    // 2. UI 이벤트 핸들러
    // ==========================================

    // 채팅창 열기/닫기
    $('.btn-chat-open, .btn-chat-close').on('click', function() {
        $('#chatWindow').toggleClass('active');

        if ($('#chatWindow').hasClass('active')) {
            $('#helpBubble').fadeOut(200);
            setTimeout(function() { $('#chatInput').focus(); }, 100);
        } else {
            $('#helpBubble').fadeIn(200);
            $('#menuDrawer').removeClass('active');
            $('#menuBtn').removeClass('active');
        }
    });

    // 메뉴 토글
    $('#menuBtn').on('click', function() {
        $('#menuDrawer').toggleClass('active');
        $(this).toggleClass('active');
    });

    // 시나리오 칩 버튼 클릭
    $('.chat-menu-wrap .btn-menu').on('click', function() {
        $('#menuDrawer').removeClass('active');
        $('#menuBtn').removeClass('active');

        let type = $(this).attr('data-type'); 

        if( type === 'event' ){
            sendMessageToUI("나의 학술행사 참가 내역을 조회해줘.", 'user');
            makeEvent();
        } else if( type === 'fee' ){
            sendMessageToUI("나의 회비 납부 내역을 확인해줘.", 'user');
            makeFee();
        } else if( type === 'active'){
            // (추가하신다면 여기에 작성)
        }
    });

    // 전송 버튼 클릭
    $('.btn-send').on('click', function() {
        processUserInput();
    });

    // 엔터키 입력
    $('#chatInput').on('keypress', function(e) {
        if (e.which == 13) { 
            processUserInput();
            return false; 
        }
    });

    // 사용자 입력 처리 함수
    function processUserInput() {
        let input = $('#chatInput');
        let message = input.val().trim();
        if (!message) return;

        input.val('').focus();
        sendMessageToUI(message, 'user'); // 화면 표시 & 저장
        streamBotResponse(message);       // 스트리밍 요청
    }

    // ==========================================
    // 3. 핵심 로직: 메시지 표시, 스트림, 저장
    // ==========================================

    /**
     * 화면에 메시지를 추가하고, 로컬스토리지에 저장하는 함수
     * @param {string} content - 원본 텍스트 (마크다운 or HTML)
     * @param {string} sender - 'user' | 'bot'
     * @param {string|null} id - 메시지 ID (옵션)
     */
    function addMessage(content, sender, id = null) {
        let displayHtml = content;

        // ★ 봇 메시지일 경우만 마크다운 변환 수행
        if (sender === 'bot') {
            // marked 라이브러리가 있으면 변환
            if (typeof marked !== 'undefined') {
                displayHtml = marked.parse(content);
            }
            // 링크 새 창 열기 처리
            displayHtml = displayHtml.replace(/<a /g, '<a target="_blank" ');
        } else {
            // 사용자 메시지는 줄바꿈만 처리
            displayHtml = content.replace(/\n/g, '<br>');
        }

        let $msgDiv = $('<div>')
            .addClass('chatbot-con ' + sender)
            .html(displayHtml); // 변환된 HTML을 삽입
        
        if (id) $msgDiv.attr('id', id);

        $('#chatBody').append($msgDiv);
        $('#chatBody').stop().animate({ scrollTop: $('#chatBody')[0].scrollHeight }, 0);
    }

    // (칩 버튼 클릭 등에서 사용) 화면 표시 + 저장 래퍼 함수
    function sendMessageToUI(text, sender) {
        addMessage(text, sender);
        saveToHistory(sender, text);
    }

    // ★ 스트리밍 요청 함수
    async function streamBotResponse(userMessage) {
        
        // 1. 로딩바 생성
        let botMsgId = 'bot-msg-' + Date.now();
        let loadingHtml = `<div class="loading"><div class="loading-icon"><div class="glow delay-0"></div><div class="glow delay-1"></div><div class="icon-circle"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="orbit-icon"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg></div></div><p>검색 중입니다...</p></div>`;
        
        // 로딩바는 저장하지 않으므로 직접 append
        let $loadingDiv = $('<div>').addClass('chatbot-con bot').attr('id', botMsgId).html(loadingHtml);
        $('#chatBody').append($loadingDiv);
        $('#chatBody').scrollTop($('#chatBody')[0].scrollHeight);

        let $botMessageDiv = $('#' + botMsgId);
        let fullBotResponse = ""; // 누적 변수 (마크다운 원본)
        let isFirstChunk = true;

        try {
            const response = await fetch('/api/chatbot/send', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
                },
                body: JSON.stringify({
                    message: userMessage,
                    thread_id: currentThreadId
                })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const jsonStr = line.replace('data: ', '').trim();
                        if (jsonStr === '[DONE]') break;
                        
                        // 스레드 ID 갱신
                        if (!jsonStr.startsWith('{') && jsonStr.startsWith('thread_')) {
                            currentThreadId = jsonStr;
                            localStorage.setItem('chat_thread_id', currentThreadId);
                            continue;
                        }

                        try {
                            const data = JSON.parse(jsonStr);
                            if (data.text) {
                                if (isFirstChunk) {
                                    $botMessageDiv.empty(); // 로딩바 제거
                                    isFirstChunk = false;   
                                }

                                // 1. 텍스트 누적 (원본)
                                fullBotResponse += data.text;

                                fullBotResponse = fullBotResponse.replace(/【.*?】/g, '');

                                // 2. 실시간 마크다운 변환 및 렌더링
                                let htmlContent = marked.parse(fullBotResponse);
                                htmlContent = htmlContent.replace(/<a /g, '<a target="_blank" ');
                                
                                // 3. 내용 갈아끼우기 (append 아님)
                                $botMessageDiv.html(htmlContent);

                                // 스크롤 유지
                                $('#chatBody').stop().animate({ scrollTop: $('#chatBody')[0].scrollHeight }, 0);
                            }
                        } catch (e) { }
                    }
                }
            }

            // ★ 스트리밍 완료 후 원본 저장
            if (fullBotResponse) {
                saveToHistory('bot', fullBotResponse);
            }

        } catch (error) {
            console.error("Stream Error:", error);
            $botMessageDiv.html("죄송합니다. 오류가 발생했습니다.");
        }
    }

    // 내역 조회 (회비/행사) 함수들
    async function makeFee() { await callApi('/api/chatbot/fee'); }
    async function makeEvent() { await callApi('/api/chatbot/event'); }

    // API 호출 공통 함수
    async function callApi(url) {
        // 로딩바 표시
        let botMsgId = 'bot-msg-' + Date.now();
        let $loadingDiv = $('<div>').addClass('chatbot-con bot').attr('id', botMsgId).html(`<div class="loading"><div class="loading-icon"><div class="glow delay-0"></div><div class="glow delay-1"></div><div class="icon-circle"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="orbit-icon"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg></div></div><p>검색 중입니다...</p></div>`);
        $('#chatBody').append($loadingDiv);
        $('#chatBody').scrollTop($('#chatBody')[0].scrollHeight);

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
                },
                body: JSON.stringify({
                    userSid: '{{ thisAuth()->check() ? encrypt(thisAuth()->user()->sid) : null }}'
                })
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const data = await response.json();
            
            // 로딩바 제거 후 결과 표시
            $('#' + botMsgId).remove();
            
            // 서버에서 받은 HTML은 이미 완성이므로 그대로 출력 및 저장
            // (addMessage가 bot이면 parse를 시도하지만, HTML 태그는 marked가 보존함)
            addMessage(data.html, 'bot');
            saveToHistory('bot', data.html);

        } catch (error) {
            console.error("요청 실패:", error);
            $('#' + botMsgId).html("정보를 불러오지 못했습니다.");
        }
    }

    // 로컬스토리지 저장 함수
    function saveToHistory(sender, content) {
        let history = JSON.parse(localStorage.getItem('chat_history')) || [];
        history.push({
            sender: sender,
            content: content, // ★ 원본 그대로 저장
            time: new Date().getTime()
        });
        localStorage.setItem('chat_history', JSON.stringify(history));
    }
});
</script>
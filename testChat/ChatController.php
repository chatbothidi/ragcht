<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;

use Illuminate\Http\Request;
use OpenAI\Laravel\Facades\OpenAI;
use Illuminate\Support\Facades\Log;

use App\Models\User;
use App\Models\Fee;
use App\Models\Registration;

class ChatController extends Controller
{
    public function sendMessage(Request $request)
    {
        $userMessage = $request->input('message');
        $threadId = $request->input('thread_id');
        $assistantId = env('OPENAI_ASSISTANT_ID');

        return response()->stream(function () use ($userMessage, $threadId, $assistantId) {
            
            // 1. 스레드 ID가 없으면 생성
            if (!$threadId) {
                $thread = OpenAI::threads()->create([]);
                $threadId = $thread->id;
                // 클라이언트에게 스레드 ID 먼저 알려줌 (특수 이벤트)
                echo "event: thread_created\n";
                echo "data: $threadId\n\n";
                flush();
            }

            // 2. 메시지 추가
            OpenAI::threads()->messages()->create($threadId, [
                'role' => 'user',
                'content' => $userMessage,
            ]);

            // 3. 스트림 실행 (createStreamed)
            $stream = OpenAI::threads()->runs()->createStreamed($threadId, [
                'assistant_id' => $assistantId,
                'additional_instructions' => "
                    현재 시각은 " . date('Y년 m월 d일 H시 i분') . "입니다. 
                    사용자가 '올해' 또는 '이번 연도'라고 말하면 무조건 " . date('Y년') . "을 기준으로 답변하세요.
                    절대로 2023년이나 과거를 현재로 착각하지 마십시오.
                "
            ]);

            foreach ($stream as $response) {
                // 4. 텍스트 델타(조각)가 들어올 때마다 전송
                if ($response->event === 'thread.message.delta') {
                    $content = $response->response->delta->content[0]->text->value ?? '';
                    
                    // 주석 기호 제거 (간단 버전) - 스트림이라 완벽하진 않지만 시도
                    // (스트림 도중에는 완벽한 정규식 제거가 어렵지만, 클라이언트에서 처리 추천)
                    
                    if (!empty($content)) {
                        // SSE 포맷: data: {내용}\n\n
                        echo "data: " . json_encode(['text' => $content]) . "\n\n";
                        
                        // PHP 버퍼 비우기 (즉시 전송)
                        if (ob_get_level() > 0) ob_flush();
                        flush();
                    }
                }
            }
            
            // 5. 종료 신호
            echo "data: [DONE]\n\n";
            flush();

        }, 200, [
            'Content-Type' => 'text/event-stream', // 필수 헤더
            'Cache-Control' => 'no-cache',
            'Connection' => 'keep-alive',
            'X-Accel-Buffering' => 'no', // Nginx 사용 시 버퍼링 끄기
        ]);
    }

    public function fee(Request $request)
    {

        $user = User::find( decrypt( $request->userSid ) );

        if( !$user ){
            return response()->json([
                'success' => false,
            ], 404);
        }

        $myFees = Fee::where(['del'=>'N','user_sid'=>$user->sid])->whereHas('user')->orderByDesc('yearv')->orderByDesc('sid')->get();

        $html = view('kr.layouts.components.chatbot.fee', ['myFees' => $myFees, 'user' => $user])->render();

        return response()->json([
            'success' => true,
            'html' => $html
        ], 200);
        
    }

    public function event(Request $request)
    {

        $user = User::find( decrypt( $request->userSid ) );

        if( !$user ){
            return response()->json([
                'success' => false,
            ], 404);
        }

        $myEvents = Registration::with('workshop')->where('usid', $user->sid)->where('del', 'N')->orderByDesc('sid')->get();

        $html = view('kr.layouts.components.chatbot.event', ['myEvents' => $myEvents, 'user' => $user])->render();

        return response()->json([
            'success' => true,
            'html' => $html
        ], 200);
        
    }
}

<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Str;

use App\Models\User;
use App\Models\Fee;
use App\Models\Registration;

class ChatController extends Controller
{
    public function sendMessage(Request $request)
    {
        $userMessage = $request->input('message');
        $threadId = $request->input('thread_id');

        // TODO: 환경변수로 url 분리하기 RAG_API_URL=218.235.94.243 분리하기 
        $ragApiUrl = env('RAG_API_URL');

        // thread_<uuid> prefix 정리 → RAG session_id 추출
        $sessionId = $threadId;
        if ($sessionId && str_starts_with($sessionId, 'thread_')) {
            $sessionId = substr($sessionId, 7);
        }
        $isNewSession = empty($sessionId);
        if ($isNewSession) {
            $sessionId = (string) Str::uuid();
        }

        return response()->stream(function () use ($userMessage, $sessionId, $ragApiUrl, $isNewSession) {

            // 1. 새 세션이면 thread_id를 클라이언트에 먼저 알림 (chat_type1.blade.php:246 매칭)
            if ($isNewSession) {
                echo "data: thread_" . $sessionId . "\n\n";
                if (ob_get_level() > 0) ob_flush();
                flush();
            }

            // 2. RAG /chat 호출 (SSE)
            $payload = json_encode([
                'query' => $userMessage,
                'session_id' => $sessionId,
                'stream' => true,
            ], JSON_UNESCAPED_UNICODE);

            $ch = curl_init($ragApiUrl . '/chat');
            curl_setopt($ch, CURLOPT_POST, true);
            curl_setopt($ch, CURLOPT_HTTPHEADER, [
                'Content-Type: application/json',
                'Accept: text/event-stream',
            ]);
            curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
            curl_setopt($ch, CURLOPT_TIMEOUT, 120);

            // 3. 청크 단위로 받아 SSE 라인 파싱 후 클라이언트로 변환·릴레이
            $buffer = '';
            curl_setopt($ch, CURLOPT_WRITEFUNCTION, function ($ch, $chunk) use (&$buffer, $ragApiUrl) {
                $buffer .= $chunk;
                while (($pos = strpos($buffer, "\n")) !== false) {
                    $line = rtrim(substr($buffer, 0, $pos), "\r");
                    $buffer = substr($buffer, $pos + 1);
                    if (!str_starts_with($line, 'data: ')) continue;
                    $raw = substr($line, 6);
                    if ($raw === '') continue;
                    $event = json_decode($raw, true);
                    if (!is_array($event)) continue;

                    if (($event['type'] ?? null) === 'token') {
                        $content = $event['content'] ?? '';
                        if ($content === '') continue;

                        // [[다운로드:파일명]] → [파일명 다운로드](RAG_URL/admin/download/encoded)
                        $content = preg_replace_callback(
                            '/\[\[다운로드:([^\]]+?)\]\]/u',
                            function ($m) use ($ragApiUrl) {
                                $name = trim($m[1]);
                                return "[{$name} 다운로드]({$ragApiUrl}/admin/download/" . rawurlencode($name) . ")";
                            },
                            $content
                        );

                        echo "data: " . json_encode(['text' => $content], JSON_UNESCAPED_UNICODE) . "\n\n";
                        if (ob_get_level() > 0) ob_flush();
                        flush();
                    }
                    // sources, done 이벤트는 무시 (DONE은 아래에서 통합 송출)
                }
                return strlen($chunk);
            });

            curl_exec($ch);
            $err = curl_error($ch);
            $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
            curl_close($ch);

            if ($err || $httpCode >= 400) {
                Log::error('RAG proxy error', ['err' => $err, 'http' => $httpCode]);
                echo "data: " . json_encode(['text' => '서버에 일시적인 문제가 있습니다. 잠시 후 다시 시도해 주세요.'], JSON_UNESCAPED_UNICODE) . "\n\n";
                if (ob_get_level() > 0) ob_flush();
                flush();
            }

            // 4. 종료 신호 (chat_type1.blade.php:243의 if (jsonStr === '[DONE]') 분기)
            echo "data: [DONE]\n\n";
            if (ob_get_level() > 0) ob_flush();
            flush();

        }, 200, [
            'Content-Type' => 'text/event-stream',
            'Cache-Control' => 'no-cache',
            'Connection' => 'keep-alive',
            'X-Accel-Buffering' => 'no',
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

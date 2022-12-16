# NEW GAME
new_game_already_exists =
   이미 진행중인 디퓨전 게임이 있습니다. 끝나면 다음 게임을 요청할 수 있어요~
new_game_should_be_direct =
   디퓨전 게임을 하려면 DM (\"멘션한 사람들만\") 으로 보내주세요. # diffuse_game 도 붙이시구요!"
new_game_prompt_is_missing =
   헉... 이미지를 만들 텍스트가 없네요!
   어떤 내용을 이미지로 만들까요? # diffuse_game 과 함께 DM 으로 보내주세요.
new_game_generation_in_progress =
   누군가가 새 디퓨전 게임을 요청해주셨어요. 어떤 그림이 나올까요?
new_game_start_announce =
   AI 그림 보고 텍스트 맞추기 게임! (30분 뒤까지)
   이 이미지는 어떤 텍스트로 만들어졌을까요? 이 글에 멘션으로 답해주세요 (영문).

   (새 게임을 시작하려면 이 게임이 끝나고 # diffuse_game 으로 DM을 보내주세요.)
   #bot_message
new_game_start_success =
   게임 생성에 성공했습니다.

# ANSWER SUBMISSION
answer_submission_game_does_not_exist =
   그치만 이 게임은 이미 끝난 걸요!
answer_submission_is_done_by_questioner =
   본인이 낸 문제를 본인이 맞출 수는 없어요. 아시잖아요~

# ANSWER SUBMISSION - SCORE
answer_submission_no_chances_left =
   아... 그게 마지막 기회였어요! 아쉽네요.
answer_submission_left_chance_many =
   {$score} 점입니다!
   아직 기회가 {$chance_count}번 남아있어요!
answer_submission_left_chance_last =
   {$score} 점입니다!
   마지막 기회가 남아있어요!
answer_submission_left_chance_none =
   {$score} 점입니다!
   마지막 기회까지 모두 사용하셨군요! 수고하셨습니다!"
answer_submission_perfect =
   {$score} 점입니다!
   정답 기준치인 {$score_early_end_condition} 점보다 높으시군요!

   축하드립니다! 👏👏👏👏👏

# GAME END
answer_submission_was_by_cw =
   문제 낸 사람과 정답
game_no_player =
   제가 기다려봤는데요, 아무도 디퓨전 게임에 참여하지 않았어요...
game_no_player_cw =
   참여한 플레이어가 없어요
game_end =
   게임 셋!
game_winner =
   이번 승자는 {$winner} 님이에요!
game_early_end =
   정답자가 나와서 게임이 일찍 끝났습니다!

# GAME WAS BY
question_by = {$account} 님이 만든 문제였어요.
gold_positive_prompt =
   프롬프트:
   {$prompt}
gold_negative_prompt =
   네거티브 프롬프트:
   {$prompt}

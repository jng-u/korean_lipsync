# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# p – 1번. ㅁ, ㅂ, ㅃ, ㅍ
# s – 2번. ㅅ, ㅆ, 
# J – 2번. ㅈ, ㅉ, ㅊ
# i – 2번/3번(ㅡ 소리의 경우). ㅑ(ㅣ+ㅏ에서 ㅣ부분), ㅡ, ㅣ, ㅢ
# a – 3번/4번(보통 강조할 때). ㅏ
# E – 3번/5번(ㅓ,ㅕ소리 위주). ㅓ, ㅐ
# o – 6번. ㅗ
# u – 7번. ㅜ, ㅘ
# t – 8번. ㄴ, ㄷ, ㄸ, ㄹ, ㅌ

# k – 3번 그러나 대부분 모음 모양을 따라감. ㄱ, ㄲ, (받침)ㅇ, ㅋ, ㅎ

def hangul_devide(korean_word):
    r_lst = []
    for w in list(korean_word.strip()):
        ## 영어인 경우 구분해서 작성함. 
        if '가'<=w<='힣':
            ## 588개 마다 초성이 바뀜. 
            ch1 = (ord(w) - ord('가'))//588
            ## 중성은 총 28가지 종류
            ch2 = ((ord(w) - ord('가')) - (588*ch1)) // 28
            ch3 = (ord(w) - ord('가')) - (588*ch1) - 28*ch2
            r_lst.append([CHOSUNG_LIST[ch1], JUNGSUNG_LIST[ch2], JONGSUNG_LIST[ch3]])
        else:
            r_lst.append([w])
    return r_lst

def jong_cho_equals(korean_word):
    w_lst = hangul_devide(korean_word)
    prev_jong = w_lst[0][2]
    for w in list(w_lst):
        if(prev_jong == w[0]) :
            w[0] = ''
            prev_jong = w[2]
    return w_lst
            

def sound_to_img(korean_word):
    w_lst = hangul_devide(korean_word)
    #print(w_lst)
    lst = []
    flag = False
    #for i, w in enumerate(w_lst):
    for w in list(w_lst):
        # w[0] 초성
        if(w[0]=='ㅁ' or w[0]=='ㅂ' or w[0]=='ㅍ'):        # shape 1
            lst.append(1)
        elif(w[0]=='ㅅ' or w[0]=='ㅆ' or w[0]=='ㅈ' or w[0]=='ㅉ' or w[0]=='ㅊ'):
            lst.append(2)
        elif(w[0]=='ㄴ' or w[0]=='ㄷ' or w[0]=='ㄹ' or w[0]=='ㄸ' or w[0]=='ㅌ'):
            lst.append(8)
        else:
            flag = True
        # w[1] 중성. 순서대로 34567에 대한 조건문
        if(w[1]=='ㅐ' or w[1]=='ㅒ' or w[1]=='ㅔ' or w[1]=='ㅖ' or w[1]=='ㅡ' or w[1]=='ㅢ' or w[1]=='ㅣ'):            # shape 3
            lst.append(3)
        elif(w[1]=='ㅏ' or w[1]=='ㅑ'):            # shape 4
            lst.append(4)
        elif(w[1]=='ㅓ' or w[1]=='ㅕ'):            # shape 5
            lst.append(5)
        elif(w[1]=='ㅗ' or w[1]=='ㅛ' or w[1]=='ㅚ' or w[1]=='ㅙ'):            # shape 6
            lst.append(6)
        elif(w[1]=='ㅘ' or w[1]=='ㅜ' or w[1]=='ㅠ' or w[1]=='ㅟ' or w[1]=='ㅝ' or w[1]=='ㅞ'):            #shape 7
            lst.append(7)   
        # w[0] == empty
        if(flag):
            lst.append(lst[len(lst)-1])
            flag = False
        # w[2] 종성
        if(w[2]=='ㅁ' or w[2]=='ㅂ' or w[2]=='ㅍ'):        # shape 1
            lst.append(1)
        elif(w[2]=='ㅅ' or w[2]=='ㅆ' or w[2]=='ㅈ' or w[2]=='ㅉ' or w[2]=='ㅊ'):
            lst.append(2)
        elif(w[2]=='ㄴ' or w[2]=='ㄷ' or w[2]=='ㄹ' or w[2]=='ㄸ' or w[2]=='ㅌ'):
            lst.append(8)
        else:
            lst.append(lst[len(lst)-1])
    #l = " ".join(map(str, lst))
    #print(l)
    return lst

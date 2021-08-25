import pandas as pd
import serial

#포트 실행-pyseiral
ard = serial.Serial(
    #포트는 실행 전에 확인
    port = 'COM4',
    baudrate = 9600
    )

#DataFrame 생성
data_v1=[[0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0]]

index_v1=['S','M','L','XL']

columns_v1=['shirt_1','shirt_2','shirt_3',
            'blouse_1','blouse_2','blouse_3',
            'hoodie_1','hoodie_2','hoodie_3']


v1=pd.DataFrame(data=data_v1, index=index_v1, columns=columns_v1)

print(v1)
print()

#창고 안의 옷들 재고 확인 및 갱신[ex) 0개->2개]
#초음파 센서의 개수에 한계가 있기 때문에 시연 시에는 센서 두개로 진행할 예정
#->for문을 두 번만 돌려 시연할 종류의 센서만 재고 파악하는 함수로 활용
def check_inventory(v1):
    distance=5
    for i in range(2):
        #distance는 아두이노 센서로 측정해야 함(나중에 변경)
        distance += 1
        

        if (distance > 20 and distance<=30):
            #재고=1
            v1.iloc[j][i]=v1.iloc[j][i]+1
                
                

        elif (distance > 10 and distance <= 20):
            #재고=2
            v1.iloc[j][i]=v1.iloc[j][i]+2
                    

        elif (distance > 0 and distance <= 10):
           #재고=3
           v1.iloc[j][i]=v1.iloc[j][i]+3
                   
               
        else:
           #shirt=0
           v1.iloc[j][i]=v1.iloc[j][i]+0
        

    
    print(v1)
    return v1


def add(v1):
    ans1=input("사이즈 입력 = ")
    ans2=input("옷 종류 입력 = ")
    ans3=int(input("개수 입력 = "))
    v1.loc[ans1, ans2]=ans3+v1.loc[ans1, ans2]
    print()

    print("옷 종류 = ", ans2, "-", ans1)
    print("재고 개수 = ",v1.loc[ans1, ans2])

    return v1

def take_clothes(v1):
    ans1=input("사이즈 입력 = ")
    ans2=input("옷 종류 입력 = ")
    ans3=int(input("개수 입력 = "))
    v1.loc[ans1, ans2]=v1.loc[ans1, ans2]-ans3
    print()

    print("옷 종류 = ", ans2, "-", ans1)
    print("재고 개수 = ",v1.loc[ans1, ans2])
    
    return v1
 

v1=check_inventory(v1)
print()

v1=v1.transpose()
print(v1)

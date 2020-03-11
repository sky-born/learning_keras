// 상수 멤버 함수
#include <iostream>

class Marine {
  static int total_marine_num;//static은 class 전체에 하나밖에 존재안함
  const static int i = 0;//const는 변하지 않음

  int hp;                // 마린 체력
  int coord_x, coord_y;  // 마린 위치
  bool is_dead;

  const int default_damage;  // 기본 공격력

 public:
  Marine();              // 기본 생성자
  Marine(int x, int y);  // x, y 좌표에 마린 생성
  Marine(int x, int y, int default_damage);

  int attack() const;                    // 데미지를 리턴한다. const function 선언 -> 함수이름뒤 const
  Marine& be_attacked(int damage_earn);  // 입는 데미지 Marine ref를 리턴함
  void move(int x, int y);               // 새로운 위치

  void show_status();  // 상태를 보여준다.
  static void show_total_marine();
  ~Marine() { total_marine_num--; }
};
int Marine::total_marine_num = 0;
void Marine::show_total_marine() {
  std::cout << "전체 마린 수 : " << total_marine_num << std::endl;
}
Marine::Marine()
    : hp(50), coord_x(0), coord_y(0), default_damage(5), is_dead(false) {
  total_marine_num++;
}

Marine::Marine(int x, int y)
    : coord_x(x),
      coord_y(y),
      hp(50),

      default_damage(5),
      is_dead(false) {
  total_marine_num++;
}

Marine::Marine(int x, int y, int default_damage)
    : coord_x(x),
      coord_y(y),
      hp(50),
      default_damage(default_damage),
      is_dead(false) {
  total_marine_num++;
}

void Marine::move(int x, int y) {
  coord_x = x;
  coord_y = y;
}
int Marine::attack() const { return default_damage; }//const 함수
Marine& Marine::be_attacked(int damage_earn) {
  hp -= damage_earn;//== this->hp -= datmage_earn;
  if (hp <= 0) is_dead = true;//== if(this->hp<=0) this->is_dead = true;

  return *this;//this pointer는 함수를 부른 객체를 지정 포인터기때문에 *을 붙여 사용
}
void Marine::show_status() {
  std::cout << " *** Marine *** " << std::endl;
  std::cout << " Location : ( " << coord_x << " , " << coord_y << " ) "
            << std::endl;
  std::cout << " HP : " << hp << std::endl;
  std::cout << " 현재 총 마린 수 : " << total_marine_num << std::endl;
}

int main() {
  Marine marine1(2, 3, 5);
  marine1.show_status();

  Marine marine2(3, 5, 10);
  marine2.show_status();

  std::cout << std::endl << "마린 1 이 마린 2 를 두 번 공격! " << std::endl;
  marine2.be_attacked(marine1.attack()).be_attacked(marine1.attack());//attacked 함수가 Marin class reference를 리턴하기 때문에 이렇게 할 수 있음

  marine1.show_status();
  marine2.show_status();
}
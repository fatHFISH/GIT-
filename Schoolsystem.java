package 项目管理系统;
import java.util.*;
public class Schoolsystem {
 static Scanner scan=new Scanner(System.in);
 int[ ] maxStudent=new int[3];    //每个班的最大人数50
 int[ ] currentStudent=new int[3];  //每个班的当前人数
 public Schoolsystem(int big, int medium, int small) {
  for(int i=0;i<3;i++) maxStudent[i]=50;
  currentStudent[0]=maxStudent[0]-big;
  currentStudent[1]=maxStudent[1]-medium;
  currentStudent[2]=maxStudent[2]-small;
 }
 public Boolean addStudent(int stuType) {
	 for(int i=1;i<=3;i++) {
		 if(stuType==i) {
		 if(maxStudent[i-1]>currentStudent[i-1]) {
			 currentStudent[i-1]++;
			 return true;
		 }
		 else break;
		 }
	 }
	 return false;
 }
 public Boolean delStudent(Integer stuType) {
	 for(int i=1;i<3;i++) {
		 if(stuType==(-i)) {
			 if(currentStudent[i-1]>0) {
				 --currentStudent[i-1];
				 System.out.println("删除"+i+"班人员成功");
				 return true;
			 }
			 
			 else break;
			 
		 }
	 }
	 return false;
 }
 

 
 public void print(boolean a) {
  System.out.print(a);
 }
 public static ArrayList<Integer> parse(String input) {
    String[] strs = input.split(" ");
    ArrayList<Integer> put=new ArrayList<Integer>();
    for(int i = 0;i < strs.length; i++)
     put.add(Integer.valueOf(strs[i]));
    return put;
    }
 public static void main(String[] args) {
   String a;
   System.out.println("分别按顺序和空格间开输入三个班级可增加人数和想要删除人员的班级以及同学要加入的班级");
      a=scan.nextLine();
      ArrayList<Integer> params=Schoolsystem.parse(a);
      System.out.print("输出结果为：");
  Schoolsystem sc=new Schoolsystem(params.get(0),params.get(1),params.get(2));
  for(int i=3;i<params.size();i++) {
	  if(params.get(i)<0) sc.delStudent(params.get(i));
  }
  for(int i=3;i<params.size();i++) {
	  if(params.get(i)>0) {
   sc.print(sc.addStudent(params.get(i)));
   System.out.print(",");
	  }
  }
  }  
}
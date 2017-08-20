#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;
int main()
{
//    Matrix3f A;
//    Vector3f b;
//    A << 1,2,3,  4,5,6,  7,8,10;
//    b << 3, 3, 4;
//    cout << "Here is the matrix A:\n" << A << endl;
//    cout << "Here is the vector b:\n" << b << endl;
//    Vector3f x = A.colPivHouseholderQr().solve(b);
//    cout << "The solution is:\n" << x << endl;

    std::vector<int> v(5);
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);

    for(int a: v) {
        std::cout << a << std::endl;
    }
    std::cout << v.size() << std::endl;
    
    return 0;
}

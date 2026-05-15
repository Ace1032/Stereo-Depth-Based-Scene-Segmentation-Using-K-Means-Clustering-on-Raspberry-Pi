#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;

static void calcBoardCorners(Size boardSize, float squareSize, vector<Point3f>& corners)
{
    corners.clear();
    for (int i = 0; i < boardSize.height; ++i)
    {
        for (int j = 0; j < boardSize.width; ++j)
        {
            corners.emplace_back(j * squareSize, i * squareSize, 0.0f);
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " imagelist.txt\n";
        cerr << "imagelist.txt should contain alternating left/right image filenames.\n";
        return 1;
    }

    const Size boardSize(7,7);      // inner corners
    const float squareSize = 0.025f; // meters, change to your real square size

    vector<string> imageList;
    {
        ifstream fs(argv[1]);
        if (!fs.is_open())
        {
            cerr << "Failed to open imagelist file: " << argv[1] << "\n";
            return 1;
        }
        string line;
        while (getline(fs, line))
        {
            if (!line.empty())
                imageList.push_back(line);
        }
    }

    if (imageList.size() < 4 || imageList.size() % 2 != 0)
    {
        cerr << "Need an even number of image filenames, at least 2 pairs.\n";
        return 1;
    }

    vector<vector<Point2f>> imagePoints1, imagePoints2;
    vector<vector<Point3f>> objectPoints;

    Size imageSize;
    vector<Point3f> boardCorners;
    calcBoardCorners(boardSize, squareSize, boardCorners);

    for (size_t i = 0; i < imageList.size(); i += 2)
    {
        Mat img1 = imread(imageList[i], IMREAD_GRAYSCALE);
        Mat img2 = imread(imageList[i + 1], IMREAD_GRAYSCALE);

        if (img1.empty() || img2.empty())
        {
            cerr << "Failed to load pair: " << imageList[i] << " and " << imageList[i + 1] << "\n";
            continue;
        }

        if (imageSize.empty())
            imageSize = img1.size();

        if (img1.size() != imageSize || img2.size() != imageSize)
        {
            cerr << "All images must have same size.\n";
            return 1;
        }

        vector<Point2f> corners1, corners2;
	
	bool found1 = findChessboardCornersSB(img1, boardSize, corners1);
	bool found2 = findChessboardCornersSB(img2, boardSize, corners2);


        if (!found1 || !found2)
        {
            cerr << "Chessboard not found in pair: " << imageList[i] << " and " << imageList[i + 1] << "\n";
            continue;
        }

        cornerSubPix(
            img1, corners1, Size(11, 11), Size(-1, -1),
            TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 30, 0.01)
        );
        cornerSubPix(
            img2, corners2, Size(11, 11), Size(-1, -1),
            TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 30, 0.01)
        );

        imagePoints1.push_back(corners1);
        imagePoints2.push_back(corners2);
        objectPoints.push_back(boardCorners);

        cout << "Accepted pair: " << imageList[i] << " , " << imageList[i + 1] << "\n";
    }

    if (imagePoints1.size() < 10)
    {
        cerr << "Not enough valid pairs. Try at least 10 good pairs.\n";
        return 1;
    }

    Mat M1, D1, M2, D2;
    Mat R, T, E, F;

    M1 = Mat::eye(3, 3, CV_64F);
    M2 = Mat::eye(3, 3, CV_64F);
    D1 = Mat::zeros(8, 1, CV_64F);
    D2 = Mat::zeros(8, 1, CV_64F);

    int flags = 0;
    flags |= CALIB_RATIONAL_MODEL;

    double rms = stereoCalibrate(
        objectPoints,
        imagePoints1,
        imagePoints2,
        M1, D1, M2, D2,
        imageSize,
        R, T, E, F,
        flags,
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5)
    );

    cout << "Stereo calibration RMS error = " << rms << "\n";

    Mat R1, R2, P1, P2, Q;
    Rect roi1, roi2;
    stereoRectify(
        M1, D1, M2, D2, imageSize,
        R, T, R1, R2, P1, P2, Q,
        CALIB_ZERO_DISPARITY, 0, imageSize, &roi1, &roi2
    );

    {
        FileStorage fs("intrinsic.yml", FileStorage::WRITE);
        fs << "M1" << M1;
        fs << "D1" << D1;
        fs << "M2" << M2;
        fs << "D2" << D2;
        fs.release();
    }

    {
        FileStorage fs("extrinsic.yml", FileStorage::WRITE);
        fs << "R"  << R;
        fs << "T"  << T;
        fs << "R1" << R1;
        fs << "R2" << R2;
        fs << "P1" << P1;
        fs << "P2" << P2;
        fs << "Q"  << Q;
        fs.release();
    }

    cout << "Wrote intrinsic.yml and extrinsic.yml\n";
    return 0;
}

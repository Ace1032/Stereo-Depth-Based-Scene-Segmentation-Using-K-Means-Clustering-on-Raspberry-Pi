#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;

static string makeName(const string& prefix, int index, const string& ext = ".png")
{
    ostringstream oss;
    oss << prefix << "_" << setfill('0') << setw(4) << index << ext;
    return oss.str();
}

int main(int argc, char** argv)
{
    int leftCam = 0;
    int rightCam = 2;
    int width = 640;
    int height = 480;
    double fps = 15.0;
    const int maxPairs = 20;

    // Chessboard pattern: 9x6 inner corners
    const Size boardSize(7, 7);

    if (argc >= 3)
    {
        leftCam = atoi(argv[1]);
        rightCam = atoi(argv[2]);
    }

    VideoCapture capL(leftCam, CAP_V4L2);
    VideoCapture capR(rightCam, CAP_V4L2);

    capL.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
    capR.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));

    if (!capL.isOpened())
    {
        cerr << "ERROR: Could not open left camera index " << leftCam << endl;
        return -1;
    }

    if (!capR.isOpened())
    {
        cerr << "ERROR: Could not open right camera index " << rightCam << endl;
        return -1;
    }

    capL.set(CAP_PROP_FRAME_WIDTH, width);
    capL.set(CAP_PROP_FRAME_HEIGHT, height);
    capL.set(CAP_PROP_FPS, fps);

    capR.set(CAP_PROP_FRAME_WIDTH, width);
    capR.set(CAP_PROP_FRAME_HEIGHT, height);
    capR.set(CAP_PROP_FPS, fps);

    width = static_cast<int>(capL.get(CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(capL.get(CAP_PROP_FRAME_HEIGHT));
    fps = capL.get(CAP_PROP_FPS);

    cout << "Left camera  : " << leftCam << '\n'
         << "Right camera : " << rightCam << '\n'
         << "Resolution   : " << width << " x " << height << '\n'
         << "FPS          : " << fps << '\n'
         << "Press 's' to save ONLY if chessboard is found in both views.\n"
         << "Press 'q' or ESC to quit.\n"
         << "Maximum pairs: " << maxPairs << '\n';

    namedWindow("left", WINDOW_AUTOSIZE);
    namedWindow("right", WINDOW_AUTOSIZE);
    namedWindow("stereo", WINDOW_NORMAL);

    int savedCount = 0;

    for (;;)
    {
        Mat frameL, frameR;
        if (!capL.read(frameL) || !capR.read(frameR))
        {
            cerr << "ERROR: Failed to read from one or both cameras.\n";
            break;
        }

        if (frameL.empty() || frameR.empty())
        {
            cerr << "ERROR: Empty frame received.\n";
            break;
        }

        Mat grayL, grayR;
        cvtColor(frameL, grayL, COLOR_BGR2GRAY);
        cvtColor(frameR, grayR, COLOR_BGR2GRAY);

        vector<Point2f> cornersL, cornersR;


	bool foundL = findChessboardCornersSB(grayL, boardSize, cornersL);
	bool foundR = findChessboardCornersSB(grayR, boardSize, cornersR);

	cout << "foundL=" << foundL << " foundR=" << foundR << endl;

        Mat displayL = frameL.clone();
        Mat displayR = frameR.clone();

        if (foundL)
        {
            cornerSubPix(
                grayL, cornersL, Size(11, 11), Size(-1, -1),
                TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 30, 0.01)
            );
            drawChessboardCorners(displayL, boardSize, cornersL, foundL);
        }

        if (foundR)
        {
            cornerSubPix(
                grayR, cornersR, Size(11, 11), Size(-1, -1),
                TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 30, 0.01)
            );
            drawChessboardCorners(displayR, boardSize, cornersR, foundR);
        }

        string status = (foundL && foundR) ? "Chessboard found in BOTH" : "Chessboard NOT ready";
        Scalar statusColor = (foundL && foundR) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);

        putText(displayL, status, Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2);
        putText(displayR, status, Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2);

        putText(displayL, "Saved: " + to_string(savedCount) + "/" + to_string(maxPairs),
                Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);
        putText(displayR, "Saved: " + to_string(savedCount) + "/" + to_string(maxPairs),
                Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);

        Mat stereoView;
        hconcat(displayL, displayR, stereoView);

        imshow("left", displayL);
        imshow("right", displayR);
        imshow("stereo", stereoView);

        int key = waitKey(1) & 0xFF;

        if (key == 27 || key == 'q')
        {
            break;
        }

/*	
        else if (key == 's')
        {
            if (!(foundL && foundR))
            {
                cout << "Skipped: chessboard not detected in both cameras.\n";
                continue;
            }

            if (savedCount >= maxPairs)
            {
                cout << "Already captured maximum of " << maxPairs << " pairs.\n";
                break;
            }

            string leftName = makeName("left", savedCount);
            string rightName = makeName("right", savedCount);

            if (!imwrite(leftName, frameL))
            {
                cerr << "ERROR: Failed to save " << leftName << '\n';
                continue;
            }

            if (!imwrite(rightName, frameR))
            {
                cerr << "ERROR: Failed to save " << rightName << '\n';
                continue;
            }

            cout << "Saved pair " << savedCount << ": "
                 << leftName << ", " << rightName << '\n';

            savedCount++;

            if (savedCount >= maxPairs)
            {
                cout << "Captured " << maxPairs << " pairs. Done.\n";
                break;
            }
        }

*/	


	else if (key == 's')
	{
    		string leftName = "left_test.png";
    		string rightName = "right_test.png";

    		imwrite(leftName, frameL);
    		imwrite(rightName, frameR);

    		cout << "Saved test pair: " << leftName << ", " << rightName << endl;
	}


    }

    capL.release();
    capR.release();
    destroyAllWindows();
    return 0;
}

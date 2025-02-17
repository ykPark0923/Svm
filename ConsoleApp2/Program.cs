using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json;
using OpenCvSharp;
using OpenCvSharp.ML;
using System.Runtime.InteropServices;

namespace ConsoleApp2
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // CSV 데이터 로드
            string fileName = "right_data.csv";
            double[,] vectors = LoadCsvData(fileName, 423, 81920); //423개의 행(샘플 수)과 81920개의 열(256*320픽셀)

            // 첫 번째 이미지 출력, vectors 데이터를 256x320 크기의 이미지로 변환하여 화면에 표시
            ShowImage(vectors, 256, 320);

            // Skeletonization 및 ROI 추출 후 데이터 가공, 81920개의 특징을 가진 데이터를 80개의 특징 벡터로 변환
            double[,] processedVectors = ProcessImages(vectors);

            // JSON에서 라벨 데이터 로드
            double[] labels = LoadLabels("right_label.json");

            // SVM 학습 및 성능 평가
            TrainAndEvaluateSVM(processedVectors, labels);
        }






        // CSV 데이터 로드
        static double[,] LoadCsvData(string filePath, int rows, int cols)
        {
            double[,] vectors = new double[rows, cols];
            string[] lines = File.ReadAllLines(filePath); //CSV 파일을 읽어서 각 행을 문자열 배열로 저장

            // CSV 파일의 데이터를 행렬로 변환
            for (int i = 0; i < rows; i++) //423행
            {
                //Select : LINQ(Language Integrated Query) 메서드로 루프필요없어 간결성과 가독성 확보(조건처리없을때만 사용)
                double[] rowData = lines[i].Split(',').Select(double.Parse).ToArray(); //i행 값 ,로 분할
                for (int j = 0; j < cols; j++)
                {
                    vectors[i, j] = rowData[j];
                }
            }
            return vectors;
        }







        // 이미지 출력
        static void ShowImage(double[,] vectors, int height, int width)
        {
            // 이미지를 저장할 공간
            Mat image = new Mat(height, width, MatType.CV_64F);  //256*320 64-bit float (double 타입)

            // 1차원 배열을 생성하여 2차원 데이터를 변환
            double[] dataArray = new double[height * width];

            //Buffer.BlockCopy를 이용해 2차원 배열을 1차원 배열로 변환
            //Buffer.BlockCopy(소스배열, 소스시작위치, 대상배열, 대상시작위치, 바이트수(256*320*8));
            //Buffer.BlockCopy 빠르고 효율적으로 데이터를 복사
            Buffer.BlockCopy(vectors, 0, dataArray, 0, height * width * sizeof(double));


            // Marshal.Copy를 이용해 1차원 배열을 OpenCV Mat 데이터로 복사
            //image.Data: OpenCV에서 이미지 데이터를 저장하는 실제 메모리 공간
            // Marshal 클래스는 .NET 메모리와 네이티브 메모리 간의 데이터 변환(마샬링, Marshaling)을 수행
            Marshal.Copy(dataArray, 0, image.Data, dataArray.Length);


            // 데이터 64-bit float → 8-bit unsigned integer
            // OpenCV에서 이미지를 표시할 때는 보통 8-bit unsigned integer(CV_8UC1) 형식을 사용
            Mat displayImage = new Mat();
            image.ConvertTo(displayImage, MatType.CV_8UC1, 255.0 / 100.0);



            /*--------------------------------------------------------------------------
             * image.ConvertTo(displayImage, MatType.CV_8UC1, scaleFactor)
             * scaleFactor = 255.0 / 100.0을 곱하여 정규화(normalization) 수행
             * OpenCV에서 CV_8UC1(8-bit unsigned integer) 포맷의 픽셀 값 범위: 0 ~ 255
             * 원본 데이터가 온도 데이터→ 픽셀 값이 0 ~ 100 범위로 존재할 수 있음
             * 따라서 0~100범위를 0~255 범위로 확장
            ---------------------------------------------------------------------------*/



            // 변환된 이미지를 화면에 표시
            Cv2.ImShow("Image", displayImage);

            // 키 입력을 대기하여 창이 바로 닫히지 않도록 함
            Cv2.WaitKey(0);

            // 모든 OpenCV 창을 닫음
            Cv2.DestroyAllWindows();
        }








        // Skeletonization 및 ROI 추출 후 데이터 가공
        static double[,] ProcessImages(double[,] vectors)
        {
            int rowCount = vectors.GetLength(0);            
            double[,] processedVectors = new double[rowCount, 80]; //이미지마다 80개의 특징 추출해 저장

            for (int j = 0; j < rowCount; j++)
            {
                Mat flirImage = new Mat(256, 320, MatType.CV_64F); //256x320 크기의 64-bit float
                double[] flirData = new double[256 * 320]; // 데이터 저장할 1차원 배열

                // Buffer.BlockCopy vertors[i][j] -> flirData[i])
                // Buffer.BlockCopy(소스배열, 소스시작위치(복사할 행의 시작점), 대상배열, 대상시작위치, 바이트수(256*320*8));
                Buffer.BlockCopy(vectors, j * 256 * 320 * sizeof(double), flirData, 0, 256 * 320 * sizeof(double));


                // Marshal.Copy를 이용해 1차원 배열을 OpenCV Mat 데이터로 복사
                // image.Data: OpenCV에서 이미지 데이터를 저장하는 실제 메모리 공간
                // Marshal 클래스는 .NET 메모리와 네이티브 메모리 간의 데이터 변환(마샬링, Marshaling)을 수행
                Marshal.Copy(flirData, 0, flirImage.Data, flirData.Length);


                // Cv2.Threshold(입력 이미지, 출력 이미지, 임계값, 최대값, 임계처리방법);
                // 픽셀 값이 41보다 크면 255(흰색), 작으면 0(검은색)으로 변환
                Mat mask = new Mat();
                Cv2.Threshold(flirImage, mask, 41, 255, ThresholdTypes.Binary);

                // 이진화된 mask 이미지를 float32(CV_32F) 형식으로 변환
                // ConvertTo(출력Mat, 변환할 데이터 타입, 스케일링 계수);
                // 픽셀 값을 0.0 ~ 1.0 범위로 조정
                mask.ConvertTo(mask, MatType.CV_32F, 1.0 / 255.0);



                Mat skeleton = Skeletonize(mask); // Skeletonization 수행

                // skeleton을 flirImage와 같은 타입으로 변환 (CV_64F)
                if (skeleton.Type() != flirImage.Type())
                {
                    skeleton.ConvertTo(skeleton, flirImage.Type());
                }



                /*----------------------------------------------------------------
                 * NaN 및 Inf 값 처리
                 * NaN(Not a Number), Inf(Infinity) 값이 포함되었는지 확인 가능
                 * Cv2.MinMaxLoc(flirImage, out double minVal, out double maxVal);
                 * Console.WriteLine($"flirImage Min: {minVal}, Max: {maxVal}");
                 * Cv2.MinMaxLoc(skeleton, out minVal, out maxVal);
                 * Console.WriteLine($"skeleton Min: {minVal}, Max: {maxVal}");
                ----------------------------------------------------------------*/



                // NaN 및 Inf 값 제거
                // NaN 및 Inf 값이 존재하면 머신러닝 모델, OpenCV 연산, 후처리 과정에서 오류 발생 가능
                // flirImage와 skeleton의 픽셀 값 중 1e9(10억)보다 큰 값 or NaN값을 1e9로 클램핑
                Cv2.Min(flirImage, new Scalar(1e9), flirImage);
                // flirImage와 skeleton의 픽셀 값 중 -1e9(-10억)보다 작은 값 or NaN값을 -1e9로 클램핑
                Cv2.Max(flirImage, new Scalar(-1e9), flirImage);
                //ThresholdTypes.Tozero : 0보다 작은 값은 0으로 변환, 음수 값제거하여 안정적인 데이터 유지
                Cv2.Threshold(flirImage, flirImage, double.MinValue, double.MaxValue, ThresholdTypes.Tozero);

                Cv2.Min(skeleton, new Scalar(1e9), skeleton);
                Cv2.Max(skeleton, new Scalar(-1e9), skeleton);
                Cv2.Threshold(skeleton, skeleton, double.MinValue, double.MaxValue, ThresholdTypes.Tozero);



                // Skeleton이 모두 0이면 기본값 설정
                // Cv2.CountNonZero() : 행렬 내 0이 아닌 픽셀의 개수를 반환
                if (Cv2.CountNonZero(skeleton) == 0)
                {
                    // skeleton이 모두 0이면 이후 연산에서 모든 값이 0이 되어 데이터 사라짐
                    Console.WriteLine("Warning: skeleton 이미지가 모두 0입니다.");
                    // Skeleton 이미지가 전부 0이면, 강제로 1.0 값을 할당, 회색으로 변경하여 연산토록 함
                    skeleton.SetTo(new Scalar(1.0));
                }

                // Skeleton과 원본 이미지 곱하기, 픽셀 단위로 곱셈연산
                // skeleton= 1.0이면 flirImage 값 유지 0.0이면 flirImage 값이 사라짐(검정색)
                Mat mixed = flirImage.Mul(skeleton);
                //Console.WriteLine($"mixed Type: {mixed.Type()}, Size: {mixed.Size()}");






                // ROI, 특징추출
                // x = 160 열부터 150픽셀, y=0 행부터 255픽셀 추출
                Mat mixedRight = mixed.SubMat(new Rect(160, 0, 150, 255));





                // 유효한 픽셀(0이 아닌 값) 위치 찾기
                List<int> res = new List<int>();
                // mixedRight의 각 행을 탐색하여 0이 아닌 픽셀의 행 인덱스를 res 리스트에 저장
                for (int i = 0; i < mixedRight.Rows; i++) // mixedRight의 모든 행을 반복하며 값 확인
                {
                    if (mixedRight.Get<double>(i, 0) > 0) // 첫번째 열(열 인덱스 0)의 픽셀 값이 0보다 크다면
                        res.Add(i); // 해당 행의 인덱스를 리스트에 추가
                }

                // 유효한 픽셀이 존재하는 경우만 특징을 추출
                if (res.Count > 0)
                {
                    // 가장 아래쪽(최대 인덱스)에 있는 유효한 픽셀의 행 번호 찾기
                    int maxn = res.Max();
                    // 최하단 유효 픽셀 위치에서 80개 위쪽까지 특징을 추출하기 위해 시작점 설정
                    int minn = maxn - 80;
                    // 80개의 특징 벡터를 저장할 배열 생성
                    double[] vector = new double[80];


                    // 특징 벡터 추출 (아래에서 위 방향으로 저장)
                    for (int i = minn, num = 0; i < maxn; i++, num++)
                    {
                        int iii = maxn - num; // 최신(가장 아래쪽) 픽셀부터 위쪽 방향으로 채우기
                        vector[num] = mixedRight.Get<double>(iii, 0); // 해당 행의 첫 번째 열 값 저장
                    }
                    // 추출한 80개의 특징 벡터를 최종 processedVectors 배열에 저장
                    for (int i = 0; i < 80; i++)
                    {
                        processedVectors[j, i] = vector[i];
                    }
                }
            }
            return processedVectors;
        }




        // Skeletonization
        static Mat Skeletonize(Mat img)
        {
            Mat skel = Mat.Zeros(img.Size(), MatType.CV_8UC1); // 결과를 저장할 빈 행렬 (0으로 초기화, 크기는 원본 이미지와 동일)
            Mat temp = new Mat(); // 임시 변수 선언
            Mat eroded = new Mat();
            Mat element = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3)); // 구조화 요소(Structuring Element) 생성 (3x3 사각형 커널)

            while (true)
            {
                /*
                 * Cv2.Erode(입력 이미지, 출력 이미지, 구조화 요소(커널(3*3이라 픽셀을 축소하는 방식으로 동작)), 반복 횟수(옵션));
                 * Erode : 작은 노이즈를 제거하거나, 윤곽선을 점진적으로 축소하는 데 사용
                 */
                Cv2.Erode(img, eroded, element); // 이미지 침식 (Erosion) → 객체의 외곽을 제거하여 크기 축소
                Cv2.Dilate(eroded, temp, element); // 침식된 이미지 팽창 (Dilation) → 객체를 원래 크기로 복원
                Cv2.Subtract(img, temp, temp); // 원본에서 팽창된 이미지를 빼서 윤곽선만 남김

                if (temp.Type() != skel.Type()) // 데이터 타입이 다를 경우 변환 (안정적인 연산을 위해 필요)
                {
                    temp.ConvertTo(temp, MatType.CV_8UC1);
                }

                Cv2.BitwiseOr(skel, temp, skel); // 윤곽선을 Skeleton 이미지에 추가 (OR 연산)
                eroded.CopyTo(img); // 원본을 침식된 이미지로 업데이트 (반복)

                if (Cv2.CountNonZero(img) == 0) // 모든 픽셀이 0이면 (검은색이면) 반복 종료
                    break;
            }
            return skel; // 결과 반환 (Skeletonized 이미지)
        }

        // JSON 파일에서 라벨데이터 로드
        static double[] LoadLabels(string filePath)
        {
            string json = File.ReadAllText(filePath); // JSON 파일을 읽어 문자열로 저장
            List<double> rawLabels = JsonConvert.DeserializeObject<List<double>>(json); // JSON 데이터를 List<double> 형식으로 변환 (리스트로 역직렬화)

            // 임계값(Threshold) 설정
            double threshold1 = 0.8; // 최소 임계값
            double threshold2 = 1.5; // 최대 임계값
            double[] labels = new double[rawLabels.Count]; // 라벨 데이터를 저장할 배열 생성

            // 모든 라벨 데이터 순회하여 변환 수행
            for (int i = 0; i < rawLabels.Count; i++)
            {
                if (rawLabels[i] < threshold1) // 값이 threshold1(0.8) 미만이면 0으로 설정
                    labels[i] = 0;
                else if (rawLabels[i] > threshold2) // 값이 threshold2(1.5) 초과이면 0으로 설정
                    labels[i] = 0;
                else
                    labels[i] = 1; // threshold1 이상 threshold2 이하 값이면 1로 설정
            }
            return labels; // 변환된 라벨 배열 반환
        }





        // SVM 학습 및 평가
        // data → 입력 데이터(특징 벡터, Feature Vector), labels → 각 데이터의 라벨(양품: 1, 불량: 0)
        static void TrainAndEvaluateSVM(double[,] data, double[] labels)
        {
            int rowCount = data.GetLength(0); // 샘플 개수(행)
            int featureCount = data.GetLength(1);  // 특징 개수(열)




            // 학습 데이터 변환
            Mat trainX = new Mat(rowCount, featureCount, MatType.CV_32F); // OpenCV Mat 객체 생성 (32-bit float)
            
            // 1차원 배열로 변환 후 trainX에 저장
            float[] trainDataArray = new float[rowCount * featureCount];
            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < featureCount; j++)
                {
                    trainDataArray[i * featureCount + j] = (float)data[i, j];
                }
            }
            trainX.SetArray(trainDataArray);





            // labels를 OpenCV Mat 형식으로 변환
            Mat trainY = new Mat(rowCount, 1, MatType.CV_32S); // OpenCV Mat 객체 생성 (32-bit 정수형)
            int[] trainLabelArray = labels.Select(l => (int)l).ToArray(); // double[] labels → int[] 변환
            trainY.SetArray(trainLabelArray);




            // SVM 모델 생성 및 학습
            SVM svm = SVM.Create();
            svm.Type = SVM.Types.CSvc; // C-Support Vector Classification (C-SVC) 분류 모델 사용
            svm.KernelType = SVM.KernelTypes.Rbf; // RBF 커널(Radial Basis Function) 사용
            svm.Degree = 3; // 다항 커널용 차수 설정 (RBF에서는 사용되지 않음)
            svm.Gamma = 1; // RBF 커널에서 데이터 분포를 조절하는 파라미터
            svm.C = 1; // 오류 허용 범위를 조절하는 정규화 파라미터 (C값이 크면 오버피팅 가능성 증가)
            svm.Coef0 = 0; // 다항 커널 및 시그모이드 커널에서 사용되는 파라미터 (RBF에서는 사용되지 않음)




            // trainX → 입력 데이터, trainY → 정답 라벨, SampleTypes.RowSample → 행 단위로 학습 데이터 제공
            if (svm.Train(trainX, SampleTypes.RowSample, trainY))
            {
                Console.WriteLine("SVM 학습 완료");
                svm.Save("SVM.xml");  // XML로 모델 저장
            }

            // 결과 예측
            Mat results = new Mat();
            svm.Predict(trainX, results); // 학습된 svm 모델을 사용하여 학습 데이터(trainX)에 대한 예측 수행

            // 데이터 타입 변환 (trainY → CV_32FC1로 변환)
            if (trainY.Type() != results.Type()) // trainY와 results의 데이터 타입이 다르면 변환
            {
                trainY.ConvertTo(trainY, results.Type()); // trainY를 results와 동일한 데이터 타입으로 변환
                //Console.WriteLine($"Converted TrainY Type: {trainY.Type()}"); // 변환 후 확인
            }

            //Console.WriteLine($"Results Size: {results.Size()}, Type: {results.Type()}");
            //Console.WriteLine($"TrainY Size: {trainY.Size()}, Type: {trainY.Type()}");


            // 정확도 계산
            Mat matches = new Mat();
            Cv2.Compare(results, trainY, matches, CmpType.EQ); // results와 trainY 비교 (같으면 1, 다르면 0)
            int correctCount = Cv2.CountNonZero(matches); // 예측 결과가 맞은 샘플 수 계산
            float accuracy = (float)correctCount / trainY.Rows * 100; // 정확도(Accuracy) 계산

            Console.WriteLine($"Accuracy: {accuracy}%");
        }
    }
}


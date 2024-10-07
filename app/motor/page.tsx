"use client";
import React, { useState } from "react";
import { FileUpload } from "@/components/ui/file-upload";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Button } from "@/components/ui/button";
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale, // Import the CategoryScale for the x-axis
    LinearScale,   // Import the LinearScale for the y-axis
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';  // Auto-imports for Line charts
import { Badge } from "@/components/ui/badge"

// Register the required components and scales
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

export default function Motor() {
    const [mode, setMode] = useState<string>("pre-rehab");
    const [selectedTrainFile, setSelectedTrainFile] = useState<File | null>(null);
    const [selectedTestFile, setSelectedTestFile] = useState<File | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const [leftHandData, setLeftHandData] = useState<number[]>([]);
    const [rightHandData, setRightHandData] = useState<number[]>([]);
    const [timeData, setTimeData] = useState<number[]>([]); // Time data
    const [accuracy, setAccuracy] = useState<number>();

    // Function to handle file selection from FileUpload component
    const handleTrainFileChange = (files: File[]) => {
        if (files.length > 0) {
            setSelectedTrainFile(files[0]);  // Assuming only one file is being uploaded
        } else {
            setSelectedTrainFile(null);
        }
    };

    // Function to handle file selection from FileUpload component
    const handleTestFileChange = (files: File[]) => {
        if (files.length > 0) {
            setSelectedTestFile(files[0]);  // Assuming only one file is being uploaded
        } else {
            setSelectedTestFile(null);
        }
    };

    // Handle form submission and POST request to FastAPI endpoint
    const handleSubmit = async () => {
        setIsLoading(true);
        if (!selectedTrainFile) {
            alert("Please upload a train file before submitting.");
            return;
        }
        if (!selectedTestFile) {
            alert("Please upload a test file before submitting.");
            return;
        }
        console.log("mode:", mode);

        // Create a FormData object to send the file
        const formData = new FormData();
        formData.append("trainfile", selectedTrainFile);
        formData.append("testfile", selectedTestFile);
        formData.append("mode", mode);  // Example of adding other form data

        console.log("selectedTrainFile:",selectedTrainFile)
        console.log("selectedTestFile:",selectedTestFile)

        try {
            const response = await fetch("http://127.0.0.1:8000/api/py/evaluate/", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Failed to submit data.");
            }

            const data = await response.json();
            console.log("Server response:", data);
            const svmLeftHandProbabilities = data.svm_left_hand_probabilities;
            const svmRightHandProbabilities = data.svm_right_hand_probabilities;
            const svmAccuracy: number = data.svm_accuracy;

            // Now you can log them or use them as needed
            console.log("SVM Left Hand Probabilities:", svmLeftHandProbabilities);
            console.log("SVM Right Hand Probabilities:", svmRightHandProbabilities);

            setLeftHandData(svmLeftHandProbabilities);
            setRightHandData(svmRightHandProbabilities);
            setTimeData(Array.from({ length: svmLeftHandProbabilities.length }, (_, i) => i));  // Simulate time axis
            setAccuracy(svmAccuracy)
            setIsLoading(false);

            return [svmLeftHandProbabilities, svmRightHandProbabilities];
            // You can handle the response here (e.g., display the results to the user)
        } catch (error) {
            console.error("Error:", error);
        }
        setIsLoading(false);
    };

    // Set up the chart data for the right hand
    const rightHandChartData = {
        labels: timeData,
        datasets: [
            {
                label: "Right Hand Probability",
                data: rightHandData,
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
            },
        ],
    };

    // Set up the chart data for the left hand
    const leftHandChartData = {
        labels: timeData,
        datasets: [
            {
                label: "Left Hand Probability",
                data: leftHandData,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
            },
        ],
    };

    return (
        <main className="flex min-h-screen flex-col items-center justify-between p-24 bg-white text-black">
            <div className="max-w-2xl mx-auto p-4">
                <h1 className="relative z-10 text-lg md:text-7xl bg-clip-text text-transparent bg-gradient-to-b from-neutral-200 to-neutral-600 text-center font-sans font-bold">
                    Stroke Rehab Data Analysis
                </h1>
                <p></p>
                <p className="text-neutral-500 max-w-lg mx-auto my-2 text-sm text-center relative z-10">
                    Here you can upload your data to get accurate predictions...
                </p>
            </div>

            {/* Radio Group to select pre-rehab or post-rehab */}
            <RadioGroup value={mode} onValueChange={setMode}>
                <div className="flex items-center space-x-2">
                    <RadioGroupItem value="pre-rehab" id="pre-rehab" />
                    <Label htmlFor="pre-rehab">Pre Rehab</Label>
                </div>
                <div className="flex items-center space-x-2">
                    <RadioGroupItem value="post-rehab" id="post-rehab" />
                    <Label htmlFor="post-rehab">Post Rehab</Label>
                </div>
            </RadioGroup>
            <div className="flex flex-row">
                {/* File upload component */}
                <FileUpload onChange={handleTrainFileChange} title="train"/>

                {/* File upload component */}
                <FileUpload onChange={handleTestFileChange} title="test" />
            </div>

            {/* Button to submit */}
            <Button variant="default" className="" onClick={handleSubmit} disabled={isLoading}>
                Get Results
            </Button>

            {accuracy && (
                <div className="py-5 justify-center items-center gap-4">
                    <Label className="pr-4">
                        Model with an Accuracy of
                    </Label>
                    <Badge>
                        {parseFloat(accuracy.toFixed(2))}%
                    </Badge>
                </div>
            )
            }
            {/* Chart for Right Hand */}
            {rightHandData.length > 0 && (
                <div className="w-full max-w-lg mt-8">
                    <h2 className="text-center mb-4 font-semibold text-xl">Right Hand Probability</h2>
                    <Line data={rightHandChartData} />
                </div>
            )}

            {/* Chart for Left Hand */}
            {leftHandData.length > 0 && (
                <div className="w-full max-w-lg mt-8">
                    <h2 className="text-center mb-4 font-semibold text-xl">Left Hand Probability</h2>
                    <Line data={leftHandChartData} />
                </div>
            )}
        </main>
    );
}

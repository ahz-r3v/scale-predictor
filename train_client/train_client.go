package main

import (
    "context"
    "fmt"
    "log"
    "os"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"

    pb "github.com/vhive-serverless/sampler/tools/train_client/grpc_client"
)


func uploadFile(client pb.ScalePredictorClient, filename string, windowSize int32) error {
    file, err := os.Open(filename)
    if err != nil {
        return fmt.Errorf("failed opening file: %v", err)
    }
    defer file.Close()

    stream, err := client.TrainByFile(context.Background())
    if err != nil {
        return fmt.Errorf("failed setting up pipe: %v", err)
    }

    buffer := make([]byte, 1024*1024) // 1MB buffer
    for {
        n, err := file.Read(buffer)
        if err != nil {
            if err.Error() == "EOF" {
                break
            }
            return fmt.Errorf("failed reading file: %v", err)
        }

        req := &pb.FileChunk{
            Filename: filename,
            WindowSize: windowSize,
            Data: buffer[:n],
        }

        if err := stream.Send(req); err != nil {
            return fmt.Errorf("failed sending data: %v", err)
        }
        time.Sleep(10 * time.Millisecond) // simulate slow upload
    }

    res, err := stream.CloseAndRecv()
    if err != nil {
        return fmt.Errorf("predictor server fault: %v", err)
    }

    fmt.Printf("predictor respond: %s\n", res.Message)
    return nil
}

func main() {
    predictor_ip_port := os.Args[2]
    file_path := os.Args[1]
    conn, err := grpc.Dial(predictor_ip_port, grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        log.Fatalf("can't connect to grpc server: %v", err)
    }
    defer conn.Close()

    client := pb.NewScalePredictorClient(conn)

    filename := file_path
    windowSize := int32(60)

    err = uploadFile(client, filename, windowSize)
    if err != nil {
        log.Fatalf("upload failed: %v", err)
    }
}

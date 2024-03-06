from GazeDetector import GazeDetector


def main():
    detector = GazeDetector()
    # detector.run_server()
    detector.run_local()



if __name__ == '__main__':
    main()
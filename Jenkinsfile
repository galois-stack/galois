pipeline{
    agent none

    options {
        timeout(time: 2, unit: 'HOURS')
    }
    triggers {
        cron('30 07 * * *') // run at 07:30
    }

    stages {
        stage('PRAJNA CI'){
            // failFast true
            parallel {
                stage('x64-linux-release') {
                    agent {
                        dockerfile {
                            label 'Sunny'
                            filename 'ubuntu_dev_jenkins.dockerfile'
                            dir 'dockerfiles'
                            // 参数由宿主主机的jenkins账号决定
                            additionalBuildArgs '''\
                            --build-arg GID=125 \
                            --build-arg UID=124 \
                            --build-arg UNAME=jenkins \
                            '''
                        }
                    }
                    environment {
                            CXX = 'clang++'
                            CC = 'clang'
                            BUILD_TYPE = 'release'
                    }
                    stages {
                        stage('env') {
                            steps {
                                sh 'uname -a'
                                sh 'echo $USER'
                                sh 'echo $PATH'
                                sh 'cmake --version'
                                sh 'clang++ --version'
                                sh 'pwd'
                                sh 'git --version'
                            }
                        }
                        stage('format') {
                            steps {
                                script {
                                    // 运行clang-format检查并将输出保存到临时文件
                                    def status = sh(script: '''
                                        find . -regex ".*\\(\\.cpp\\|\\.h\\|\\.hpp\\|\\.cxx\\)" -exec clang-format --dry-run --verbose {} \\; > clang_format_output.txt 2>&1
                                        exit 0
                                    ''', returnStatus: true)

                                    // 读取输出文件，提取不符合规范的文件
                                    def output = readFile('clang_format_output.txt').trim()
                                    def nonCompliantFiles = []
                                    output.eachLine { line ->
                                        if (line.contains('non-compliant')) {
                                            def fileMatch = (line =~ /Formatting\s(.+)/)
                                            if (fileMatch) {
                                                nonCompliantFiles << fileMatch[0][1]
                                            }
                                        }
                                    }

                                    // 根据结果输出提示信息
                                    if (nonCompliantFiles) {
                                        echo "警告：以下文件不符合clang-format规范，请运行 'clang-format -i <file>' 修复："
                                        nonCompliantFiles.each { file ->
                                            echo "- ${file}"
                                        }
                                    } else {
                                        echo "所有文件均符合clang-format规范。"
                                    }

                                    // 清理临时文件
                                    sh 'rm -f clang_format_output.txt'
                                }
                            }
                        }
                        stage('build') {
                            steps {
                                sh './scripts/clone_submodules.sh -f --jobs=4 --depth=50'
                                sh './scripts/configure.sh ${BUILD_TYPE} -DPRAJNA_WITH_JUPYTER=OFF -DPRAJNA_DISABLE_ASSERTS=ON'
                                sh './scripts/build.sh ${BUILD_TYPE} install'
                            }
                        }
                        stage('test') {
                            steps {
                                sh './scripts/test.sh ${BUILD_TYPE}'
                            }
                        }
                        // stage("leak-check") {
                        //     steps {
                        //         sh 'valgrind --leak-check=full  --num-callers=10 --trace-children=yes ./scripts/test.sh ${BUILD_TYPE}'
                        //     }
                        // }
                    }
                }

                stage('aarch64-osx-release') {
                    agent {
                        label 'Mac'
                    }
                    environment {
                        CXX = 'clang++'
                        CC = 'clang'
                        BUILD_TYPE = 'release'
                    }
                    stages {
                        stage('env') {
                            steps {
                                sh 'echo $USER'
                                sh 'uname -a'
                                sh 'cmake --version'
                                sh 'clang++ --version'
                                sh 'pwd'
                                sh 'git --version'
                                sh 'git config --global --list'
                            }
                        }
                        stage('format') {
                            steps {
                                script {
                                    // 运行clang-format检查并将输出保存到临时文件
                                    def status = sh(script: '''
                                        find . -regex ".*\\(\\.cpp\\|\\.h\\|\\.hpp\\|\\.cxx\\)" -exec clang-format --dry-run --verbose {} \\; > clang_format_output.txt 2>&1
                                        exit 0
                                    ''', returnStatus: true)

                                    // 读取输出文件，提取不符合规范的文件
                                    def output = readFile('clang_format_output.txt').trim()
                                    def nonCompliantFiles = []
                                    output.eachLine { line ->
                                        if (line.contains('non-compliant')) {
                                            def fileMatch = (line =~ /Formatting\s(.+)/)
                                            if (fileMatch) {
                                                nonCompliantFiles << fileMatch[0][1]
                                            }
                                        }
                                    }

                                    // 根据结果输出提示信息
                                    if (nonCompliantFiles) {
                                        echo "警告：以下文件不符合clang-format规范，请运行 'clang-format -i <file>' 修复："
                                        nonCompliantFiles.each { file ->
                                            echo "- ${file}"
                                        }
                                    } else {
                                        echo "所有文件均符合clang-format规范。"
                                    }

                                    // 清理临时文件
                                    sh 'rm -f clang_format_output.txt'
                                }
                            }
                        }
                        stage('build') {
                            steps {
                                sh './scripts/clone_submodules.sh -f --jobs=4 --depth=50'
                                sh './scripts/configure.sh ${BUILD_TYPE} -DPRAJNA_WITH_JUPYTER=OFF -DPRAJNA_DISABLE_ASSERTS=ON'
                                sh './scripts/build.sh ${BUILD_TYPE} install'
                            }
                        }
                        stage('test') {
                            steps {
                                sh './scripts/test.sh ${BUILD_TYPE} --gtest_filter=-*gpu*'
                            }
                        }
                        // stage("leak-check") {
                        //     steps {
                        //         sh 'valgrind --leak-check=full  --num-callers=10 --trace-children=yes ./scripts/test.sh ${BUILD_TYPE}'
                        //     }
                        // }
                    }
                }
            }
        }
    }
}

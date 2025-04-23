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
                                sh 'echo "Running clang-format check..."'
                                sh './scripts/check_clang_format.sh'
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
                                sh 'echo "Running clang-format check..."'
                                sh './scripts/check_clang_format.sh'
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

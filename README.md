import Foundation
import MetalKit

// ----------------------------------------------------------
// MARK: - NEURAL SIMULATION CORE
// ----------------------------------------------------------

struct Neuron {
    var voltage: Float
    var threshold: Float
    var decay: Float
    var connections: [Int]
}

class NeuralNetwork {
    private(set) var neurons: [Neuron]
    private var metalDevice: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var computePipeline: MTLComputePipelineState!

    init(neuronCount: Int) {
        neurons = (0..<neuronCount).map { _ in
            Neuron(
                voltage: Float.random(in: -1...1),
                threshold: Float.random(in: 0.1...0.6),
                decay: Float.random(in: 0.92...0.98),
                connections: []
            )
        }

        for i in 0..<neuronCount {
            neurons[i].connections = (0..<20).map { _ in Int.random(in: 0..<neuronCount) }
        }

        setupMetal()
    }

    // Initialize Metal GPU acceleration
    func setupMetal() {
        metalDevice = MTLCreateSystemDefaultDevice()
        commandQueue = metalDevice.makeCommandQueue()

        let shader = """
        kernel void computeNeurons(
            device float *voltage [[buffer(0)]],
            device float *threshold [[buffer(1)]],
            device float *decay [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            float v = voltage[id];
            v *= decay[id];
            if (v > threshold[id]) { v = -0.5; }
            voltage[id] = v;
        }
        """

        let library = try! metalDevice.makeLibrary(source: shader, options: nil)
        let function = library.makeFunction(name: "computeNeurons")!
        computePipeline = try! metalDevice.makeComputePipelineState(function: function)
    }

    func tick() {
        let count = neurons.count
        let voltages = neurons.map { $0.voltage }
        let thresholds = neurons.map { $0.threshold }
        let decays = neurons.map { $0.decay }

        let vBuffer = metalDevice.makeBuffer(bytes: voltages, length: MemoryLayout<Float>.size * count)!
        let tBuffer = metalDevice.makeBuffer(bytes: thresholds, length: MemoryLayout<Float>.size * count)!
        let dBuffer = metalDevice.makeBuffer(bytes: decays, length: MemoryLayout<Float>.size * count)!

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(computePipeline)
        encoder.setBuffer(vBuffer, offset: 0, index: 0)
        encoder.setBuffer(tBuffer, offset: 0, index: 1)
        encoder.setBuffer(dBuffer, offset: 0, index: 2)

        let threads = MTLSize(width: count, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threads, threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let updatedPointer = vBuffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { neurons[i].voltage = updatedPointer[i] }
    }
}

// ----------------------------------------------------------
// MARK: - EXPERIMENT RUNNER
// ----------------------------------------------------------

class NeuralExperiment {
    private let network: NeuralNetwork
    private var steps: Int

    init(neurons: Int, steps: Int) {
        self.network = NeuralNetwork(neuronCount: neurons)
        self.steps = steps
    }

    func start() {
        print("Running neural simulation with \(network.neurons.count) neurons…")
        for step in 0..<steps {
            network.tick()
            if step % 100 == 0 {
                let avg = network.neurons.map { $0.voltage }.reduce(0, +) / Float(network.neurons.count)
                print("Step \(step): Avg Voltage = \(avg)")
            }
        }
    }
}

// ----------------------------------------------------------
// MARK: - MAIN PROGRAM
// ----------------------------------------------------------

let experiment = NeuralExperiment(neurons: 5000, steps: 5000)
experiment.start()
